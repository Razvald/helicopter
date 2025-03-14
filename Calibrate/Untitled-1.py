# %%
import os
import csv
import cv2
import json
import math
from time import time
from tqdm import tqdm
from itertools import product

# %%
import vio_ort_exp as vio_ort

# %%
# Инициализация параметров
odometry = vio_ort.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
set_dir = '2025_3_14_11_42_57_num_4'

json_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.json')])
start = 0
count_json = len(json_files)

# %%
# Значения для параметров
top_k_values = [512, 256]
detection_threshold_values = [0.05, 0.01]
max_iters_values = [None, 100, 300, 500]
rotation_methods = ["PIL", "CV2"]
trace_values = [8, 4]

# Генерация всех комбинаций
parameters = [
    {'top_k': top_k, 'detection_threshold': detection_threshold, 'maxIters': max_iters, 'rotation': rotation, 'trace': trace}
    for top_k, detection_threshold, max_iters, rotation, trace in product(
        top_k_values, detection_threshold_values, max_iters_values, rotation_methods, trace_values
    )
]

# %%
def run_vio(odometry, json_files, start, count_json, top_k, detection_threshold, maxIters, rotation, trace):
    """
    Выполняет обработку данных с использованием заданных параметров.
    """
    lat_VIO, lon_VIO = [], []

    odometry._matcher.top_k = top_k
    odometry._matcher.detection_threshold = detection_threshold
    odometry.MAX_ITERS = maxIters
    odometry.ROTATION = rotation
    odometry.TRACE = trace

    for filename in json_files[start:start + count_json]:
        with open(f'{set_dir}/{filename}', 'r') as file:
            data = json.load(file)
            img_path = os.path.join(set_dir, os.path.splitext(filename)[0] + '.jpg')
            image = cv2.imread(img_path)
            result_vio = odometry.add_trace_pt(image, data)

            lat_VIO.append(result_vio['lat'])
            lon_VIO.append(result_vio['lon'])

    return {
        'lat_VIO': lat_VIO,
        'lon_VIO': lon_VIO,
    }

# %%
def haversine(lat1, lon1, lat2, lon2):
    """
    Функция для вычисления расстояния между двумя точками на поверхности Земли.
    """
    R = 6371000  # радиус Земли в метрах
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# %%
def calculate_aggregated_metrics(results):
    """
    Функция для вычисления метрик траектории (среднее, максимальное, минимальное расстояние).
    """
    metrics = []

    for i in range(1, len(results['lat_VIO'])):
        dist = haversine(
            results['lat_VIO'][0], results['lon_VIO'][0],
            results['lat_VIO'][i], results['lon_VIO'][i]
        )
        metrics.append(dist)

    mean_distance = sum(metrics) / len(metrics)
    max_distance = max(metrics)
    min_distance = min(metrics)
    rmse = math.sqrt(sum(d ** 2 for d in metrics) / len(metrics))

    return {
        "Mean Distance": mean_distance,
        "Max Distance": max_distance,
        "Min Distance": min_distance,
        "RMSE": rmse,
    }

# %%
def transform_vio_coords(vio_lon_list, vio_lat_list):
    """
    Преобразование координат VIO в нормализованные координаты.
    """
    vio_lon0 = vio_lon_list[0]
    vio_lat0 = vio_lat_list[0]

    vio_lon_range = max(vio_lon_list) - min(vio_lon_list)
    vio_lat_range = max(vio_lat_list) - min(vio_lat_list)

    scale_for_lon = 1 / vio_lat_range  # Широта -> долгота
    scale_for_lat = 1 / vio_lon_range  # Долгота -> широта

    transformed_lon = [(v_lat - vio_lat0) * scale_for_lon + vio_lon0 for v_lat in vio_lat_list]
    transformed_lat = [-(v_lon - vio_lon0) * scale_for_lat + vio_lat0 for v_lon in vio_lon_list]

    return transformed_lon, transformed_lat

# %%
def save_results_to_csv(results_all, filename):
    """
    Сохраняет результаты эксперимента в CSV файл.
    """
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Заголовок
        writer.writerow(["Top_k", "Detection Threshold", "Max Iters", "Rotation method", "Trace depth", "Time", "Mean Distance", "Max Distance", "RMSE"])

        # Данные
        for result in results_all:
            params = result['params']
            metrics = calculate_aggregated_metrics(result['results'])

            writer.writerow([
                params['top_k'], params['detection_threshold'], params['maxIters'] if params['maxIters'] is not None else 'None', params['rotation'], params['trace'],
                result['time'],
                metrics["Mean Distance"],  # Среднее расстояние
                metrics["Max Distance"],   # Максимальное расстояние
                metrics["RMSE"],           # RMSE
            ])

# %%
# Запуск экспериментов с разными параметрами
results_all = []
output_dir = 'output_graphs'  # Папка для графиков
os.makedirs(output_dir, exist_ok=True)

with tqdm(total=len(parameters), desc="Processing experiments", unit="experiment") as pbar:
    for params in parameters:
        start_time = time()
        results = run_vio(odometry, json_files, start, count_json, **params)
        elapsed_time = time() - start_time

        # Трансформация координат
        results['lon_VIO_transformed'], results['lat_VIO_transformed'] = transform_vio_coords(
            results['lon_VIO'], results['lat_VIO']
        )

        results_all.append({'params': params, 'results': results, 'time': elapsed_time})

        # Построение и сохранение графика
        #save_trajectory_plot(results, params, output_dir)
        pbar.update(1)

# Сохраняем результаты
save_results_to_csv(results_all, "vio_results_no_gps.csv")
