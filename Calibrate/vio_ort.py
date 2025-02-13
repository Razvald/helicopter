import json
from time import time
from datetime import datetime, date, timedelta

import numpy as np
import cv2
from PIL import Image

from modules.xfeat_ort import XFeat

from pymavlink import mavutil

import nvtx


# Загрузка параметров камеры
with nvtx.annotate("Load Camera Parameters", color="darkgreen"):
    with open('fisheye_2024-09-18.json') as f:
        camparam = json.load(f)

    MASK = None
    for shape in camparam['shapes']:
        if shape['label'] == 'mask':
            with nvtx.annotate("Create Camera Mask", color="purple"):
                MASK = np.zeros((camparam['imageHeight'], camparam['imageWidth'], 3), dtype=np.uint8)
                cnt = np.asarray(shape['points']).reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(MASK, [cnt], -1, (255, 255, 255), -1)

    with nvtx.annotate("Initialize Camera Constants", color="brown"):
        CENTER = [camparam['ppx'], camparam['ppy']]
        CENTER[0] += -6  # TODO: insert corrections into file
        CENTER[1] += 26  # TODO: insert corrections into file
        FOCAL = camparam['focal']
        RAD = camparam['radius']
        CROP_CENTER = np.asarray([RAD / 2, RAD / 2])


# Константы
HOMO_THR = 2.0
NUM_MATCH_THR = 8
TRACE_DEPTH = 4
VEL_FIT_DEPTH = TRACE_DEPTH
METERS_DEG = 111320

FLAGS = mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY

class VIO():
    
    def __init__(self, lat0=0, lon0=0, alt0=0, top_k=512, detection_threshold=0.05):
        self.lat0 = lat0
        self.lon0 = lon0
        self._matcher = XFeat(top_k=top_k, detection_threshold=detection_threshold)
        self.track = []
        self.trace = []
        self.prev = None
        self.HoM = None
        
    @nvtx.annotate("VIO.add_trace_pt", color="blue")
    def add_trace_pt(self, frame, msg):
        
        # Получаем углы и высоту – можно обернуть в отдельный NVTX-блок
        
        with nvtx.annotate("VIO.add_trace_pt.Fetch Angles & Height", color="green"):
            angles = fetch_angles(msg)
            height = fetch_height(msg)
            timestamp = time()
        
        with nvtx.annotate("VIO.add_trace_pt.Preprocess Frame", color="yellow"):
            frame = preprocess_frame(frame, MASK)

        with nvtx.annotate("VIO.add_trace_pt.Rotate Image", color="orange"):
            roll, pitch = angles['roll'] / np.pi * 180, angles['pitch'] / np.pi * 180
            dpp = (int(CENTER[0] + roll * 2.5),
                   int(CENTER[1] + pitch * 2.5))
            rotated = Image.fromarray(frame).rotate(angles['yaw']/np.pi*180, center=dpp)
            rotated = np.asarray(rotated)

        # Здесь можно добавить аннотацию для remapping и дальнейшей обработки
        with nvtx.annotate("VIO.add_trace_pt.Remap and Crop", color="cyan"):
            map_x, map_y = fisheye2rectilinear(FOCAL, dpp, RAD, RAD)
            crop = cv2.remap(rotated, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Далее обработка трассы, расчёт позиции и т.д.
        with nvtx.annotate("VIO.add_trace_pt.Detect and compute", color="red"):
            trace_pt = dict(crop=crop,
                            out=self.detect_and_compute(crop),
                            angles=angles,
                            height=height)
        
        if len(self.trace)>TRACE_DEPTH:
            self.trace = self.trace[1:]

        # Например, аннотировать расчёт локальной позиции:
        with nvtx.annotate("VIO.add_trace_pt.Calculate Local Position", color="magenta"):
            if len(self.trace) == 0:
                trace_pt['local_posm'] = np.asarray([0, 0])
            else:
                local_pos_metric = self.calc_pos(trace_pt)
                if local_pos_metric is None:
                    trace_pt['local_posm'] = self.trace[-1]['local_posm']
                else:
                    trace_pt['local_posm'] = local_pos_metric
        
        self.trace.append(trace_pt)
        self.track.append(np.hstack((timestamp, trace_pt['local_posm'], height)))

        # Пример расчёта скорости
        with nvtx.annotate("VIO.add_trace_pt.Fit Velocity", color="darkorange"):
            ts, tn, te, he = np.asarray(self.track[-TRACE_DEPTH:]).T
            if len(tn) >= TRACE_DEPTH:
                vn = np.polyfit(ts, tn, 1)[0]
                ve = np.polyfit(ts, te, 1)[0]
                vd = 0
            else:
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
            next_pt['out'],)

            if len(match_prev) <= NUM_MATCH_THR:
                continue
            
            center_proj = cv2.perspectiveTransform(CROP_CENTER.reshape(-1,1,2), HoM).ravel()
            pix_shift = CROP_CENTER - center_proj
            pix_shift[0], pix_shift[1] = -pix_shift[1], pix_shift[0]
            height = np.mean([prev_pt['height'], next_pt['height']])
            metric_shift = pix_shift / FOCAL * height
            local_pos = prev_pt['local_posm'] + metric_shift
            poses.append(local_pos)

        if len(poses):
            return np.mean(poses, axis=0)
        else:
            return None
    
    def match_points_hom(self, out0, out1):
        idxs0, idxs1 = self._matcher.match(out0['descriptors'], out1['descriptors'], min_cossim=-1 )
        mkpts_0 = out0['keypoints'][idxs0].numpy()
        mkpts_1 = out1['keypoints'][idxs1].numpy()
        
        good_prev = []
        good_next = []
        if len(mkpts_0) >= NUM_MATCH_THR:
            HoM, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, HOMO_THR)

            mask = mask.ravel().astype(bool)

            good_prev = mkpts_0[mask]
            good_next = mkpts_1[mask]
            return good_prev, good_next, HoM
        else:
            return [], [], np.eye(3)

    def detect_and_compute(self, frame):
        img = self._matcher.parse_input(frame)
        out = self._matcher.detectAndCompute(img)[0]
        return out

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
    #angles = msg['EXT_IMU']
    angles['yaw'] = -angles['yaw']
    return angles

def fetch_height(msg):
    return max(0, msg['AHRS2']['altitude'])

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