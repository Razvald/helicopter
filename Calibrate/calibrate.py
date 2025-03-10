import os
import csv
import cv2
import json
import math
import numpy as np
from time import time
import matplotlib.pyplot as plt

import vio_ort_exp as vio_ort
import vio_ort_org as vio_ort_original

# Инициализация параметров
odometry = vio_ort.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
odometry_org = vio_ort_original.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
set_dir = '2024_12_15_15_31_8_num_3'
json_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.json')])
start = 1000
count_json = 1000

def run_vio(odometry, json_files, start, count_json):
    lat_VIO, lon_VIO, alt_VIO = [], [], []
    lat_GPS, lon_GPS, alt_GPS = [], [], []
    for filename in json_files[start:start + count_json]:
        with open(f'{set_dir}/{filename}', 'r') as file:
            data = json.load(file)
            if 'GNRMC' in data and data['GNRMC']['status'] == 'A':
                img_path = os.path.join(set_dir, os.path.splitext(filename)[0] + '.jpg')
                image = cv2.imread(img_path)
                result_vio = odometry.add_trace_pt(image, data)
                lat_VIO.append(result_vio['lat'])
                lon_VIO.append(result_vio['lon'])
                alt_VIO.append(result_vio['alt'] * 1000)
                lat_GPS.append(data['GNRMC'].get('lat', 0.0))
                lon_GPS.append(data['GNRMC'].get('lon', 0.0))
                alt_GPS.append(data['GPS_RAW_INT']['alt'])
    return {
        'lat_VIO': lat_VIO,
        'lon_VIO': lon_VIO,
        'alt_VIO': alt_VIO,
        'lat_GPS': lat_GPS,
        'lon_GPS': lon_GPS,
        'alt_GPS': alt_GPS,
    }

start_time = time()
#run_vio(odometry, json_files, start, count_json)
elapsed_time = time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

start_time = time()
results_optimized = run_vio(odometry, json_files, start, count_json)
opt_elapsed_time = time() - start_time
print(f"Elapsed time: {opt_elapsed_time:.2f} seconds")

start_time = time()
results_original = run_vio(odometry_org, json_files, start, count_json)
org_elapsed_time = time() - start_time
print(f"Elapsed time: {org_elapsed_time:.2f} seconds")

def transform_vio_coords(vio_lon_list, vio_lat_list, gps_lon_list, gps_lat_list):
    gps_lon0 = gps_lon_list[0]
    gps_lat0 = gps_lat_list[0]
    vio_lon0 = vio_lon_list[0]
    vio_lat0 = vio_lat_list[0]

    gps_lon_range = max(gps_lon_list) - min(gps_lon_list)
    gps_lat_range = max(gps_lat_list) - min(gps_lat_list)
    vio_lon_range = max(vio_lon_list) - min(vio_lon_list)
    vio_lat_range = max(vio_lat_list) - min(vio_lat_list)

    scale_for_lon = gps_lon_range / vio_lat_range  # VIO широта -> GPS долгота
    scale_for_lat = gps_lat_range / vio_lon_range  # VIO долгота -> GPS широта

    transformed_lon = [(v_lat - vio_lat0) * scale_for_lon + gps_lon0 for v_lat in vio_lat_list]
    transformed_lat = [-(v_lon - vio_lon0) * scale_for_lat + gps_lat0 for v_lon in vio_lon_list]

    return transformed_lon, transformed_lat

# Применяем трансформацию
transformed_lon_opt, transformed_lat_opt = transform_vio_coords(
    results_optimized['lon_VIO'], results_optimized['lat_VIO'],
    results_optimized['lon_GPS'], results_optimized['lat_GPS']
)

transformed_lon_org, transformed_lat_org = transform_vio_coords(
    results_original['lon_VIO'], results_original['lat_VIO'],
    results_original['lon_GPS'], results_original['lat_GPS']
)

# Обновляем результаты для построения графиков
results_optimized['lon_VIO_transformed'] = transformed_lon_opt
results_optimized['lat_VIO_transformed'] = transformed_lat_opt
results_original['lon_VIO_transformed'] = transformed_lon_org
results_original['lat_VIO_transformed'] = transformed_lat_org

# Вывод последних координат для проверки
print("Last GPS coordinates:")
print(f"Latitude: {results_optimized['lat_GPS'][-1]}")
print(f"Longitude: {results_optimized['lon_GPS'][-1]}")
print("Last VIO original coordinates:")
print(f"Latitude: {results_original['lat_VIO'][-1]}")
print(f"Longitude: {results_original['lon_VIO'][-1]}")
print("Last VIO original transformed coordinates:")
print(f"Latitude: {results_original['lat_VIO_transformed'][-1]}")
print(f"Longitude: {results_original['lon_VIO_transformed'][-1]}")
print("Last VIO optimized coordinates:")
print(f"Latitude: {results_optimized['lat_VIO'][-1]}")
print(f"Longitude: {results_optimized['lon_VIO'][-1]}")
print("Last VIO optimized transformed coordinates:")
print(f"Latitude: {results_optimized['lat_VIO_transformed'][-1]}")
print(f"Longitude: {results_optimized['lon_VIO_transformed'][-1]}")

# Функция для построения графика с GPS и VIO
def plot_comparison(results_optimized, results_original):
    vio_lat_org = results_original['lat_VIO']
    vio_lon_org = results_original['lon_VIO']
    vio_alt_org = results_original['alt_VIO']

    vio_lat_opt = results_optimized['lat_VIO']
    vio_lon_opt = results_optimized['lon_VIO']
    vio_alt_opt = results_optimized['alt_VIO']

    gps_lat = results_original['lat_GPS']
    gps_lon = results_original['lon_GPS']
    gps_alt = results_original['alt_GPS']

    vio_lat_transformed_opt = results_optimized['lat_VIO_transformed']
    vio_lon_transformed_opt = results_optimized['lon_VIO_transformed']

    vio_lat_transformed_org = results_original['lat_VIO_transformed']
    vio_lon_transformed_org = results_original['lon_VIO_transformed']

    # Построение графиков широты
    plt.figure(figsize=(18, 6))
    plt.subplot(2, 3, 1)
    plt.plot(vio_lat_org, label='Original VIO Latitude', linestyle='--')
    plt.plot(vio_lat_opt, label='Optimized VIO Latitude', linestyle='-.')
    plt.plot(gps_lat, label='GPS Latitude', linestyle='-')
    plt.xlabel('Index')
    plt.ylabel('Latitude')
    plt.title('Latitude Comparison')
    plt.legend()

    # Построение графиков долготы
    plt.subplot(2, 3, 2)
    plt.plot(vio_lon_org, label='Original VIO Longitude', linestyle='--')
    plt.plot(vio_lon_opt, label='Optimized VIO Longitude', linestyle='-.')
    plt.plot(gps_lon, label='GPS Longitude', linestyle='-')
    plt.xlabel('Index')
    plt.ylabel('Longitude')
    plt.title('Longitude Comparison')
    plt.legend()

    # Построение графиков высоты
    plt.subplot(2, 3, 3)
    plt.plot(vio_alt_org, label='Original VIO Altitude', linestyle='--')
    plt.plot(vio_alt_opt, label='Optimized VIO Altitude', linestyle='-.')
    plt.plot(gps_alt, label='GPS Altitude', linestyle='-')
    plt.xlabel('Index')
    plt.ylabel('Altitude (mm)')
    plt.title('Altitude Comparison')
    plt.legend()

    # Построение широты
    plt.subplot(2, 3, 4)
    plt.plot(gps_lat, label='GPS Latitude', linestyle='-')
    plt.plot(vio_lat_transformed_org, label='Original VIO Latitude (transformed)', linestyle='--')
    plt.plot(vio_lat_transformed_opt, label='Optimized VIO Latitude (transformed)', linestyle='-.')
    plt.xlabel('Index')
    plt.ylabel('Latitude')
    plt.title('Latitude Comparison (Transformed)')
    plt.legend()

    # Построение долготы
    plt.subplot(2, 3, 5)
    plt.plot(gps_lon, label='GPS Longitude', linestyle='-')
    plt.plot(vio_lon_transformed_org, label='Original VIO Longitude (transformed)', linestyle='--')
    plt.plot(vio_lon_transformed_opt, label='Optimized VIO Longitude (transformed)', linestyle='-.')
    plt.xlabel('Index')
    plt.ylabel('Longitude')
    plt.title('Longitude Comparison (Transformed)')
    plt.legend()

    # Построение высоты
    plt.subplot(2, 3, 6)
    plt.plot(gps_alt, label='GPS Altitude', linestyle='-')
    plt.plot(vio_alt_org, label='Original VIO Altitude', linestyle='--')
    plt.plot(vio_alt_opt, label='Optimized VIO Altitude', linestyle='-.')
    plt.xlabel('Index')
    plt.ylabel('Altitude (mm)')
    plt.title('Altitude Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

#plot_comparison(results_optimized, results_original)

# Функция для вычисления расстояния между двумя точками в метрах
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # радиус Земли в метрах
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def save_results_to_csv(filename, results_optimized, results_original, comment):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Comment", "Index", "GPS Latitude", "GPS Longitude",
                "Original Latitude", "Original Longitude", "Diff GPS-Orig (m)",
                "Original Transformed Latitude", "Original Transformed Longitude", "Diff GPS-Orig Trans (m)",
                "Optimized Latitude", "Optimized Longitude", "Diff GPS-Opt (m)",
                "Optimized Transformed Latitude", "Optimized Transformed Longitude", "Diff GPS-Opt Trans (m)"
            ])

        for i in range(len(results_original['lat_GPS'])):
            # Координаты
            gps_lat, gps_lon = results_original['lat_GPS'][i], results_original['lon_GPS'][i]
            org_lat, org_lon = results_original['lat_VIO'][i], results_original['lon_VIO'][i]
            org_trans_lat, org_trans_lon = results_original['lat_VIO_transformed'][i], results_original['lon_VIO_transformed'][i]
            opt_lat, opt_lon = results_optimized['lat_VIO'][i], results_optimized['lon_VIO'][i]
            opt_trans_lat, opt_trans_lon = results_optimized['lat_VIO_transformed'][i], results_optimized['lon_VIO_transformed'][i]

            # Расчёт расстояний
            diff_gps_org = haversine(gps_lat, gps_lon, org_lat, org_lon)
            diff_gps_org_trans = haversine(gps_lat, gps_lon, org_trans_lat, org_trans_lon)
            diff_gps_opt = haversine(gps_lat, gps_lon, opt_lat, opt_lon)
            diff_gps_opt_trans = haversine(gps_lat, gps_lon, opt_trans_lat, opt_trans_lon)

            # Запись строки
            writer.writerow([
                comment, i, gps_lat, gps_lon,
                org_lat, org_lon, diff_gps_org,
                org_trans_lat, org_trans_lon, diff_gps_org_trans,
                opt_lat, opt_lon, diff_gps_opt,
                opt_trans_lat, opt_trans_lon, diff_gps_opt_trans
            ])

comment = "1000. Change PIL to CV2"
csv_filename = "vio_comparison_results.csv"
save_results_to_csv(csv_filename, results_optimized, results_original, comment)
print(f"Results appended to {csv_filename}.")
