import json
from time import time
from datetime import datetime, date, timedelta

import numpy as np
import cv2
from PIL import Image

from modules.xfeat_ort import XFeat

from pymavlink import mavutil


with open('fisheye_2024-09-18.json') as f:
    camparam = json.load(f)

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
TRACE_DEPTH = 4
VEL_FIT_DEPTH = TRACE_DEPTH
METERS_DEG = 111320

FLAGS = mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY

def count_none_recursive(arr):
    count = 0
    for item in arr:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            count += count_none_recursive(item)
        elif item is None:
            count += 1
    return count

def pt2h(abs_pressure, temperature, P0):
    return (1 - abs_pressure/P0) * 8.3144598 * (273.15 + temperature/100) / 9.80665 / 0.0289644

class VIO():

    def __init__(self, lat0=0, lon0=0, alt0=0, top_k=512, detection_threshold=0.05):
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

        angles= fetch_angles(msg)
        height = self.fetch_height(msg)
        timestamp = time()

        frame = preprocess_frame(frame, MASK)

        roll, pitch = angles['roll'] / np.pi * 180, angles['pitch'] / np.pi * 180

        dpp = (int(CENTER[0] + roll * 2.5),
                 int(CENTER[1] + pitch * 2.5)
        )

        rotated = Image.fromarray(frame).rotate(angles['yaw']/np.pi*180, center=dpp)
        rotated = np.asarray(rotated)

        map_x, map_y = fisheye2rectilinear(FOCAL, dpp, RAD, RAD)
        crop = cv2.remap(rotated, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        trace_pt = dict(crop=crop,
                        out= self.detect_and_compute(crop),
                        angles=angles,
                        height=height,
                       )

        if len(self.trace)>TRACE_DEPTH:
            self.trace = self.trace[1:]

        if len(self.trace)==0:
            trace_pt['local_posm'] = np.asarray([0, 0])
        else:
            local_pos_metric = self.calc_pos(trace_pt)
            if local_pos_metric is None:
                # copy previous value if no one matches found on any of the previous frames
                trace_pt['local_posm'] = self.trace[-1]['local_posm']
            else:
                trace_pt['local_posm'] = local_pos_metric

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
        lon = self.lon0 + te[-1] / 111320 / np.cos(self.lat0/180*np.pi) # used lat0 to avoid problems with wrong calculated latitude
        alt = he[-1]
        GPS_week, GPS_ms = calc_GPS_week_time()

        return dict(timestamp=float(ts[-1]),
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
            match_prev, match_next, HoM = self.match_points_hom(prev_pt['out'],
                                                                next_pt['out'],
                                                                )

            if len(match_prev) <= NUM_MATCH_THR:
                continue

            center_proj = cv2.perspectiveTransform(CROP_CENTER.reshape(-1,1,2), HoM).ravel()
            pix_shift = CROP_CENTER - center_proj
            pix_shift[0], pix_shift[1] = -pix_shift[1], pix_shift[0]
            height = np.mean([prev_pt['height'], next_pt['height']])
            ############################################
            ##### умножать
            metric_shift = pix_shift / FOCAL * height
            #########################################
            local_pos = prev_pt['local_posm'] + metric_shift
            poses.append(local_pos)

        if len(poses):
            return np.mean(poses, axis=0)
        else:
            return None

    def match_points_hom(self, out0, out1):
        idxs0, idxs1 = self._matcher.match(out0['descriptors'], out1['descriptors'], min_cossim=-1 )
        mkpts_0, mkpts_1 = out0['keypoints'][idxs0].numpy(), out1['keypoints'][idxs1].numpy()

        good_prev = []
        good_next = []
        if len(mkpts_0)>=NUM_MATCH_THR:
            HoM, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, HOMO_THR)

            mask = mask.ravel()
            good_prev = np.asarray([pt for ii, pt in enumerate(mkpts_0) if mask[ii]])
            good_next = np.asarray([pt for ii, pt in enumerate(mkpts_1) if mask[ii]])

            return good_prev, good_next, HoM

        else:
            return [], [], np.eye(3)

    def detect_and_compute(self, frame):
        img = self._matcher.parse_input(frame)
        out = self._matcher.detectAndCompute(img)[0]
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

    def vio2pixhawk(self, msg):

        viom = msg['VIO']

        return  [int(viom['timestamp']*10**6), # Timestamp (micros since boot or Unix epoch)
                0, # GPS sensor id in th, e case of multiple GPS
                FLAGS, # flags to ignore 8, 16, 32 etc
                # (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY) |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY,

                viom['GPS_ms'], # GPS time (milliseconds from start of GPS week)
                viom['GPS_week'], # GPS week number
                3, # 0-1: no fix, 2: 2D fix, 3: 3D fix. 4: 3D with DGPS. 5: 3D with RTK
                int(viom['lat']*10**7), # Latitude (WGS84), in degrees * 1E7
                int(viom['lon']*10**7), # Longitude (WGS84), in degrees * 1E7
                viom['alt'], # Altitude (AMSL, not WGS84), in m (positive for up)
                1.0, # GPS HDOP horizontal dilution of precision in m
                1.0, # GPS VDOP vertical dilution of precision in m
                viom['veln'], # GPS velocity in m/s in NORTH direction in earth-fixed NED frame
                viom['vele'], # GPS velocity in m/s in EAST direction in earth-fixed NED frame
                viom['veld'], # GPS velocity in m/s in DOWN direction in earth-fixed NED frame
                0.6, # GPS speed accuracy in m/s
                5.0, # GPS horizontal accuracy in m
                3.0, # GPS vertical accuracy in m
                10, # Number of satellites visible,
               ]

def calc_GPS_week_time():
    today = date.today()
    now = datetime.now()
    epoch = date(1980, 1, 6)

    epochMonday = epoch - timedelta(epoch.weekday())
    todayMonday = today - timedelta(today.weekday())
    GPS_week = int((todayMonday - epochMonday).days / 7)
    GPS_ms = ((today - todayMonday).days * 24 + now.hour) * 3600000 + now.minute*60000 + now.second*1000 + int(now.microsecond/1000)
    return GPS_week, GPS_ms

def fetch_angles(msg):
    angles = msg['ATTITUDE']
    angles['yaw'] = -angles['yaw']
    return angles


def extract_neighborhood(image, keypoint, size):
    x, y = keypoint
    half_size = size // 2

    x_start = x - half_size
    x_end = x + half_size
    y_start = y - half_size
    y_end = y + half_size
    # Reject keypoints too close to boundaries
    if x_start<0 or x_end>image.shape[1] or y_start<0 or y_end>image.shape[0]:
        return None

    nbh = image[y_start:y_end, x_start:x_end]
    # Reject keypoints  with mask pixels
    if np.any(nbh==0):
        return None
    else:
        return nbh

def fisheye2rectilinear(focal, pp, rw, rh, fproj='equidistant'):
    # Create a grid for the rectilinear image
    rx, ry = np.meshgrid(np.arange(rw) - rw // 2, np.arange(rh) - rh // 2)
    r = np.sqrt(rx**2 + ry**2) / focal

    angle_n = np.arctan(r)
    if fproj == 'equidistant':
        angle_n = angle_n
    elif fproj == 'orthographic':
        angle_n = np.sin(angle_n)
    elif fproj == 'stereographic':
        angle_n = 2*np.tan(angle_n/2)
    elif fproj == 'equisolid':
        angle_n = 2*np.sin(angle_n/2)

    angle_t = np.arctan2(ry, rx)

    pt_x = focal * angle_n * np.cos(angle_t) + pp[0]
    pt_y = focal * angle_n * np.sin(angle_t) + pp[1]

    map_x = pt_x.astype(np.float32)
    map_y = pt_y.astype(np.float32)

    return map_x, map_y

def preprocess_frame(frame, mask):
    frame = np.where(mask, frame, 0)
    return frame
