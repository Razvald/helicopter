import json
from time import time
from datetime import datetime, date, timedelta

import numpy as np
import cv2

from modules.xfeat_ort import XFeat

from pymavlink import mavutil

from nvtx import start_range, end_range

# Load camera parameters from JSON file
with open('fisheye_2024-09-18.json') as f:
    camparam = json.load(f)

# Create a mask for the image
for shape in camparam['shapes']:
    if shape['label']=='mask':
        MASK = np.zeros((camparam['imageHeight'], camparam['imageWidth'], 3), dtype=np.uint8)
        cnt = np.asarray(shape['points']).reshape(-1,1,2).astype(np.int32)
        cv2.drawContours(MASK, [cnt], -1, (255,255,255), -1)

CENTER = [camparam['ppx'], camparam['ppy']]
CENTER[0] += -6 #TODO insert corrections into file
CENTER[1] += 26 #TODO insert corrections into file
FOCAL = camparam['focal']
RAD = camparam['radius']
CROP_CENTER = np.asarray([RAD/2, RAD/2])

HOMO_THR = 2.0
NUM_MATCH_THR = 8
TRACE_DEPTH = 2
VEL_FIT_DEPTH = TRACE_DEPTH
METERS_DEG = 111320

FLAGS = mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY

class VIO():
    def __init__(self, lat0=0, lon0=0, alt0=0, top_k=256, detection_threshold=0.01):
        self.lat0 = lat0
        self.lon0 = lon0
        self._matcher = XFeat(top_k=top_k, detection_threshold=detection_threshold)
        self.track = []
        self.trace = []
        self.prev = None
        self.HoM = None
        self.height = 0
        self.P0 = None

    def add_trace_pt(self, frame, msg):
        range_id_fetch = start_range("VIO.add_trace_pt.Fetch Angles & Height", color="green")
        angles = fetch_angles(msg)
        height = self.fetch_height(msg)
        timestamp = time()
        end_range(range_id_fetch)

        range_id_preprocess = start_range("VIO.add_trace_pt.Preprocess Frame", color="yellow")
        frame = preprocess_frame(frame, MASK)
        end_range(range_id_preprocess)

        range_id_rotate = start_range("VIO.add_trace_pt.Rotate Image", color="orange")
        roll, pitch = angles['roll'] / np.pi * 180, angles['pitch'] / np.pi * 180

        dpp = (int(CENTER[0] + roll * 2.5), 
               int(CENTER[1] + pitch * 2.5)
        )

        M = cv2.getRotationMatrix2D(dpp, angles['yaw'] / np.pi * 180, 1)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        rotated = np.asarray(rotated)
        end_range(range_id_rotate)

        range_id_remap = start_range("VIO.add_trace_pt.Remap and Crop", color="cyan")
        map_x, map_y = fisheye2rectilinear(FOCAL, dpp, RAD, RAD)
        crop = cv2.remap(rotated, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        end_range(range_id_remap)

        range_id_detect = start_range("VIO.add_trace_pt.Detect and compute", color="red")
        trace_pt = dict(crop=crop,
                        out= self.detect_and_compute(crop),
                        angles=angles,
                        height=height,
                        )
        end_range(range_id_detect)

        if len(self.trace) > TRACE_DEPTH:
            self.trace = self.trace[1:]

        range_id_calc = start_range("VIO.add_trace_pt.Calculate Local Position", color="green")
        if len(self.trace)==0:
            trace_pt['local_posm'] = np.asarray([0, 0])
        else:
            local_pos_metric = self.calc_pos(trace_pt)
            if local_pos_metric is None:
                # copy previous value if no one matches found on any of the previous frames
                trace_pt['local_posm'] = self.trace[-1]['local_posm']
            else:
                trace_pt['local_posm'] = local_pos_metric
        end_range(range_id_calc)

        self.trace.append(trace_pt)
        self.track.append(np.hstack((timestamp, trace_pt['local_posm'], height)))

        ts, tn, te, he = np.asarray(self.track[-VEL_FIT_DEPTH:]).T
        if len(tn)>=VEL_FIT_DEPTH:
            # enough data to calculate velocity
            vn = np.polyfit(ts, tn, 1)[0]
            ve = np.polyfit(ts, te, 1)[0]
            vd = 0 #- np.polyfit(ts, he, 1)[0]
        else:
            # set zero velocity if data insufficient
            vn, ve, vd = 0, 0, 0

        lat = self.lat0 + tn[-1] / METERS_DEG
        lon = self.lon0 + te[-1] / 111320 / np.cos(self.lat0 / 180 * np.pi) # used lat0 to avoid problems with wrong calculated latitude
        alt = he[-1]
        GPS_week, GPS_ms = calc_GPS_week_time()

        return dict(
            timestamp=float(ts[-1]),
            to_north=float(tn[-1]),
            to_east=float(te[-1]),
            lat=float(lat),
            lon=float(lon),
            alt=float(alt),
            veln=float(vn),
            vele=float(ve),
            veld=float(vd),
            GPS_week=int(GPS_week),
            GPS_ms=int(GPS_ms)
        )

    def calc_pos(self, next_pt):
        poses = []
        for prev_pt in self.trace:
            range_id_match = start_range("VIO.calc_pos.Match Points Hom", color="green")
            match_prev, match_next, HoM = self.match_points_hom(prev_pt['out'], next_pt['out'],)
            end_range(range_id_match)

            if len(match_prev) <= NUM_MATCH_THR:
                continue

            range_id_center = start_range("VIO.calc_pos.Center Projection", color="black")
            center_proj = cv2.perspectiveTransform(CROP_CENTER.reshape(-1, 1, 2), HoM).ravel()
            pix_shift = CROP_CENTER - center_proj
            pix_shift[0], pix_shift[1] = -pix_shift[1], pix_shift[0]
            height = np.mean([prev_pt['height'], next_pt['height']])
            metric_shift = pix_shift / FOCAL * height
            local_pos = prev_pt['local_posm'] + metric_shift
            poses.append(local_pos)
            end_range(range_id_center)

        return np.mean(poses, axis=0) if poses else None

    def match_points_hom(self, out0, out1):
        range_id_match = start_range("VIO.match_points_hom", color="blue")
        idxs0, idxs1 = self._matcher.match(out0['descriptors'], out1['descriptors'], min_cossim=-1)
        mkpts_0, mkpts_1 = out0['keypoints'][idxs0].numpy(), out1['keypoints'][idxs1].numpy()
        end_range(range_id_match)

        good_prev = []
        good_next = []

        range_id_hom = start_range("VIO.match_points_hom.Homography", color="green")
        if len(mkpts_0) >= NUM_MATCH_THR:
            HoM, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, HOMO_THR, maxIters=500)
            end_range(range_id_hom)

            range_id_filter = start_range("VIO.match_points_hom.Filter Matches", color="red")
            mask = mask.ravel()
            good_prev = mkpts_0[mask.astype(bool)]
            good_next = mkpts_1[mask.astype(bool)]
            end_range(range_id_filter)
            return good_prev, good_next, HoM
        else:
            return [], [], np.eye(3)

    def detect_and_compute(self, frame):
        range_id_parse = start_range("VIO.detect_and_compute.Parse Input", color="blue")
        img = self._matcher.parse_input(frame)
        end_range(range_id_parse)

        range_id_detect = start_range("VIO.detect_and_compute.Detect and Compute", color="red")
        out = self._matcher.detectAndCompute(img)[0]
        end_range(range_id_detect)
        return out

    def fetch_height(self, msg):
        if self.P0 == None:
            self.P0 = msg['SCALED_PRESSURE']['press_abs']
        if self.P0 != None:
            self.height = pt2h(
                msg['SCALED_PRESSURE']['press_abs'],
                msg['SCALED_PRESSURE']['temperature'],
                self.P0
            )
        pres =  msg['SCALED_PRESSURE']['press_abs']
        temp = msg['SCALED_PRESSURE']['temperature']
        #print(f'height: {height}, {pres}, {temp}, {self.P0}', end='\t\r')
        return max(0, self.height)

def pt2h(abs_pressure, temperature, P0):
    return (1 - abs_pressure / P0) * 8.3144598 * (273.15 + temperature / 100) / 9.80665 / 0.0289644

def calc_GPS_week_time():
    today = date.today()
    now = datetime.now()
    epoch = date(1980, 1, 6)
    epochMonday = epoch - timedelta(epoch.weekday())
    todayMonday = today - timedelta(today.weekday())
    GPS_week = int((todayMonday - epochMonday).days / 7)
    GPS_ms = ((today - todayMonday).days * 24 + now.hour) * 3600000 + now.minute * 60000 + now.second * 1000 + int(now.microsecond / 1000)
    return GPS_week, GPS_ms

def fetch_angles(msg):
    angles = msg['ATTITUDE']
    angles['yaw'] = -angles['yaw']
    return angles

def fisheye2rectilinear(focal, pp, rw, rh, fproj='equidistant'):
    # Create a grid for the rectilinear image
    rx, ry = np.meshgrid(np.arange(rw) - rw // 2, np.arange(rh) - rh // 2)
    r = np.sqrt(rx ** 2 + ry ** 2) / focal

    angle_n = np.arctan(r)
    if fproj == 'equidistant':
        angle_n = angle_n
    elif fproj == 'orthographic':
        angle_n = np.sin(angle_n)
    elif fproj == 'stereographic':
        angle_n = 2 * np.tan(angle_n / 2)
    elif fproj == 'equisolid':
        angle_n = 2 * np.sin(angle_n / 2)

    angle_t = np.arctan2(ry, rx)

    pt_x = focal * angle_n * np.cos(angle_t) + pp[0]
    pt_y = focal * angle_n * np.sin(angle_t) + pp[1]

    map_x = pt_x.astype(np.float32)
    map_y = pt_y.astype(np.float32)

    return map_x, map_y

def preprocess_frame(frame, mask):
    return np.where(mask, frame, 0)