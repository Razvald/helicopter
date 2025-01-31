import json
from time import time
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import numpy as np
import cv2
from modules.xfeat_ort import XFeat
from pymavlink import mavutil

# Константы
HOMO_THR = 2.0
NUM_MATCH_THR = 8
TRACE_DEPTH = 4
VEL_FIT_DEPTH = TRACE_DEPTH
METERS_DEG = 111320
FLAGS = mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT | \
        mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY | \
        mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY

# Вспомогательные функции
def load_camera_params(filename='fisheye_2024-09-18.json'):
    """Загружает параметры камеры из файла JSON."""
    with open(filename) as f:
        return json.load(f)

def create_mask(camparam):
    """Создает маску на основе параметров камеры."""
    for shape in camparam['shapes']:
        if shape['label'] == 'mask':
            mask = np.zeros((camparam['imageHeight'], camparam['imageWidth'], 3), dtype=np.uint8)
            cnt = np.asarray(shape['points']).reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
            return mask
    return None

# Инициализация камеры
camparam = load_camera_params()
MASK = create_mask(camparam)
CENTER = [camparam['ppx'] - 6, camparam['ppy'] + 26]
FOCAL = camparam['focal']
RAD = camparam['radius']
CROP_CENTER = np.asarray([RAD / 2, RAD / 2])

class VIO():
    def __init__(self, lat0=0, lon0=0, alt0=0, top_k=512, detection_threshold=0.05):
        self.lat0 = lat0
        self.lon0 = lon0
        self._matcher = XFeat(top_k=top_k, detection_threshold=detection_threshold)
        self.track = []
        self.trace = []
        self.prev = None
        self.HoM = None

    def add_trace_pt(self, frame, msg):
        """Добавляет точку трассировки и возвращает результаты."""
        total_start_time = time()
        timings = {}

        # Этапы обработки
        time_local = time()
        angles = fetch_angles(msg)
        timings['fetch_angles'] = time() - time_local

        time_local = time()
        height = fetch_height(msg)
        timings['fetch_height'] = time() - time_local

        time_local = time()
        frame = preprocess_frame(frame, MASK)
        timings['preprocess_frame'] = time() - time_local

        time_local = time()
        roll, pitch = angles['roll'] / np.pi * 180, angles['pitch'] / np.pi * 180
        dpp = (int(CENTER[0] + roll * 2.5), int(CENTER[1] + pitch * 2.5))
        (h, w) = frame.shape[:2]
        M = cv2.getRotationMatrix2D(dpp, angles['yaw'] / np.pi * 180, 1.0)
        rotated = cv2.warpAffine(frame, M, (w, h))
        timings['rotation'] = time() - time_local

        time_local = time()
        map_x, map_y = fisheye2rectilinear(FOCAL, dpp, RAD, RAD)
        crop = cv2.remap(rotated, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        timings['fisheye_correction'] = time() - time_local

        time_local = time()
        trace_pt = dict(crop=crop, out=self.detect_and_compute(crop), angles=angles, height=height)
        timings['detect_and_compute'] = time() - time_local

        # Обновление позиции
        time_local = time()
        if len(self.trace) > TRACE_DEPTH:
            self.trace = self.trace[1:]

        if len(self.trace) == 0:
            trace_pt['local_posm'] = np.asarray([0, 0])
        else:
            local_pos_metric = self.calc_pos(trace_pt)
            trace_pt['local_posm'] = local_pos_metric if local_pos_metric is not None else self.trace[-1]['local_posm']

        self.trace.append(trace_pt)
        self.track.append(np.hstack((time(), trace_pt['local_posm'], height)))
        timings['local_position_calculation'] = time() - time_local

        # Скорость
        time_local = time()
        ts, tn, te, he = np.asarray(self.track[-VEL_FIT_DEPTH:]).T
        if len(tn) >= VEL_FIT_DEPTH:
            vn = (tn[-1] - tn[0]) / (ts[-1] - ts[0])
            ve = (te[-1] - te[0]) / (ts[-1] - ts[0])
            vd = 0
        else:
            vn, ve, vd = 0, 0, 0
        timings['velocity_calculation'] = time() - time_local

        time_local = time()
        lat = self.lat0 + tn[-1] / METERS_DEG
        lon = self.lon0 + te[-1] / 111320 / np.cos(self.lat0 / 180 * np.pi)
        alt = he[-1]
        GPS_week, GPS_ms = calc_GPS_week_time()
        timings['GPS_calculation'] = time() - time_local

        elapsed_time = time() - total_start_time
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
                    GPS_ms=int(GPS_ms), 
                    elapsed_time=elapsed_time, 
                    timings=timings)

    def calc_pos(self, next_pt):
        """Рассчитывает локальную позицию."""
        recent_trace = self.trace[-TRACE_DEPTH:]

        def process_prev_pt(prev_pt, next_pt, match_points_hom):
            match_prev, match_next, HoM = match_points_hom(prev_pt['out'], next_pt['out'])
            if len(match_prev) <= NUM_MATCH_THR:
                return None
            center_proj = cv2.perspectiveTransform(CROP_CENTER.reshape(-1, 1, 2), HoM).ravel()
            pix_shift = CROP_CENTER - center_proj
            pix_shift[0], pix_shift[1] = -pix_shift[1], pix_shift[0]
            return prev_pt['local_posm'] + pix_shift / FOCAL * next_pt['height']

        with Pool() as pool:
            poses = pool.map(process_prev_pt, recent_trace)
        return np.mean([p for p in poses if p is not None], axis=0) if poses else None

    def match_points_hom(self, out0, out1):
        """Находит гомографию между двумя наборами точек."""
        idxs0, idxs1 = self._matcher.match(out0['descriptors'], out1['descriptors'], min_cossim=-1)
        mkpts_0, mkpts_1 = out0['keypoints'][idxs0].numpy(), out1['keypoints'][idxs1].numpy()
        if len(mkpts_0) >= NUM_MATCH_THR:
            HoM, mask = cv2.findHomography(mkpts_0, mkpts_1, cv2.RANSAC, HOMO_THR)
            return mkpts_0[mask.ravel() == 1], mkpts_1[mask.ravel() == 1], HoM
        return [], [], np.eye(3)

    def detect_and_compute(self, frame):
        """Распознает ключевые точки и их дескрипторы."""
        img = self._matcher.parse_input(frame)
        return self._matcher.detectAndCompute(img)[0]

# Вспомогательные функции
def calc_GPS_week_time():
    """Вычисляет номер недели GPS и время в миллисекундах с начала недели."""
    today = date.today()
    now = datetime.now()
    epoch = date(1980, 1, 6)
    epochMonday = epoch - timedelta(epoch.weekday())
    todayMonday = today - timedelta(today.weekday())
    GPS_week = int((todayMonday - epochMonday).days / 7)
    GPS_ms = ((today - todayMonday).days * 24 + now.hour) * 3600000 + now.minute * 60000 + now.second * 1000 + int(now.microsecond / 1000)
    return GPS_week, GPS_ms

def fetch_angles(msg):
    """Получает углы из сообщения."""
    angles = msg['ATTITUDE']
    angles['yaw'] = -angles['yaw']
    return angles

def fetch_height(msg):
    """Получает высоту из сообщения."""
    return max(0, msg['AHRS2']['altitude'])

def fisheye2rectilinear(focal, pp, rw, rh, fproj='equidistant'):
    """Приводит изображение из фишай-проекции к прямолинейной."""
    rx, ry = np.meshgrid(np.arange(rw) - rw // 2, np.arange(rh) - rh // 2)
    r = np.sqrt(rx**2 + ry**2) / focal
    angle_n = np.arctan(r) if fproj == 'equidistant' else None
    angle_t = np.arctan2(ry, rx)
    pt_x = focal * angle_n * np.cos(angle_t) + pp[0]
    pt_y = focal * angle_n * np.sin(angle_t) + pp[1]
    return pt_x.astype(np.float32), pt_y.astype(np.float32)

def preprocess_frame(frame, mask):
    """Применяет маску к изображению."""
    return np.where(mask, frame, 0)