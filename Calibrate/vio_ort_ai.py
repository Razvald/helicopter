import json
import numpy as np
import cv2
from modules.xfeat_ort import XFeat
from pymavlink import mavutil
from datetime import datetime, date, timedelta
from time import time

# Load camera parameters from JSON file
with open('fisheye_2024-09-18.json') as f:
    camparam = json.load(f)

# Create a mask for the image
MASK = np.zeros((camparam['imageHeight'], camparam['imageWidth'], 3), dtype=np.uint8)
for shape in camparam['shapes']:
    if shape['label'] == 'mask':
        cnt = np.asarray(shape['points']).reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(MASK, [cnt], -1, (255, 255, 255), -1)

# Calculate camera parameters
CENTER = np.asarray([camparam['ppx'] - 6, camparam['ppy'] + 26])  #TODO insert corrections into file
FOCAL = camparam['focal']
RAD = camparam['radius']
CROP_CENTER = np.asarray([RAD / 2, RAD / 2])

HOMO_THR = 2.0
NUM_MATCH_THR = 8
TRACE_DEPTH = 2
VEL_FIT_DEPTH = TRACE_DEPTH
METERS_DEG = 111320

FLAGS = mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY

class VIO:
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
        angles = fetch_angles(msg)
        height = self.fetch_height(msg)
        timestamp = time()

        frame = preprocess_frame(frame, MASK)

        roll, pitch = angles['roll'] / np.pi * 180, angles['pitch'] / np.pi * 180
        dpp = (int(CENTER[0] + roll * 2.5), int(CENTER[1] + pitch * 2.5))
        M = cv2.getRotationMatrix2D(dpp, angles['yaw'] / np.pi * 180, 1)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        rotated = np.asarray(rotated)

        map_x, map_y = fisheye2rectilinear(FOCAL, dpp, RAD, RAD)
        crop = cv2.remap(rotated, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        trace_pt = dict(
            crop=crop,
            out = self.detect_and_compute(crop),
            angles=angles,
            height=height,
        )

        if len(self.trace) > TRACE_DEPTH:
            self.trace = self.trace[1:]

        if len(self.trace) == 0:
            trace_pt['local_posm'] = np.zeros(2)
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
        vn = np.polyfit(ts, tn, 1)[0] if len(tn) >= VEL_FIT_DEPTH else 0
        ve = np.polyfit(ts, te, 1)[0] if len(te) >= VEL_FIT_DEPTH else 0
        vd = 0 #- np.polyfit(ts, he, 1)[0]

        lat = self.lat0 + tn[-1] / METERS_DEG
        lon = self.lon0 + te[-1] / (111320 * np.cos(self.lat0 / 180 * np.pi))
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
            match_prev, match_next, HoM = self.match_points_hom(prev_pt['out'], next_pt['out'])
            if len(match_prev) <= NUM_MATCH_THR:
                continue

            # Получение проекции центра
            center_proj = cv2.perspectiveTransform(CROP_CENTER.reshape(-1, 1, 2), HoM).ravel()

            # Вычисление смещения пикселей
            pix_shift = CROP_CENTER - center_proj
            pix_shift = pix_shift[::-1] * [-1, 1]

            # Рассчитываем высоту как среднее между предыдущей и текущей
            height = np.mean([prev_pt['height'], next_pt['height']])

            # Рассчитываем смещение в метриках
            metric_shift = pix_shift / FOCAL * height
            local_pos = prev_pt['local_posm'] + metric_shift
            poses.append(local_pos)

        return np.mean(poses, axis=0) if poses else None

    def match_points_hom(self, out0, out1):
        idxs0, idxs1 = self._matcher.match(out0['descriptors'], out1['descriptors'], min_cossim=-1)
        mkpts_0, mkpts_1 = out0['keypoints'][idxs0].numpy(), out1['keypoints'][idxs1].numpy()
        good_prev, good_next = [], []
        if len(mkpts_0) >= NUM_MATCH_THR:
            HoM, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, HOMO_THR, maxIters=500)
            mask = mask.ravel()
            good_prev = mkpts_0[mask.astype(bool)]
            good_next = mkpts_1[mask.astype(bool)]
            return good_prev, good_next, HoM
        else:
            return [], [], np.eye(3)

    def detect_and_compute(self, frame):
        img = self._matcher.parse_input(frame)
        out = self._matcher.detectAndCompute(img)[0]
        return out

    def fetch_height(self, msg):
        if self.P0 is None:
            self.P0 = msg['SCALED_PRESSURE']['press_abs']
        pres = msg['SCALED_PRESSURE']['press_abs']
        temp = msg['SCALED_PRESSURE']['temperature']
        self.height = max(0, pt2h(pres, temp, self.P0))
        return self.height

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