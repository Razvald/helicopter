# %%
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
from time import time

import vio_ort as vio_ort
import vio_ort_org as vio_ort_original
# %%
# Инициализация глобальных параметров
odometry = vio_ort.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
odometry_org = vio_ort_original.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
set_dir = '2024_12_15_15_31_8_num_3'
json_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.json')])
start = 1000
count_json = 700
lat_VIO, lon_VIO, alt_VIO = [], [], []
lat_GPS, lon_GPS, alt_GPS = [], [], []

# %%
def run_vio(odometry, json_files, start, count_json):
    lat_VIO, lon_VIO, alt_VIO = [], [], []
    lat_GPS, lon_GPS, alt_GPS = [], [], []

    for filename in json_files[start:start + count_json]:
        with open(f'{set_dir}/{filename}', 'r') as file:
            data = json.load(file)
            if 'GNRMC' in data:
                if data['GNRMC']['status'] == 'A':
                    img_path = set_dir + '/' + os.path.splitext(filename)[0] + '.jpg'
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
# %%
timer = time()
results_test_cache = run_vio(odometry, json_files, start, count_json)
print(f"Test start for cache: {time() - timer:.2f} seconds")

timer = time()
results_optimized = run_vio(odometry, json_files, start, count_json)
print(f"Execution time for opt: {time() - timer:.2f} seconds")

timer = time()
results_original = run_vio(odometry_org, json_files, start, count_json)
print(f"Execution time for org: {time() - timer:.2f} seconds")
# %%
def calculate_errors(results):
    lat_diff = np.array(results['lat_VIO']) - np.array(results['lat_GPS'])
    lon_diff = np.array(results['lon_VIO']) - np.array(results['lon_GPS'])
    alt_diff = np.array(results['alt_VIO']) - np.array(results['alt_GPS'])

    lat_rmse = np.sqrt(np.mean(lat_diff**2))
    lon_rmse = np.sqrt(np.mean(lon_diff**2))
    alt_rmse = np.sqrt(np.mean(alt_diff**2))

    return {
        'lat_rmse': lat_rmse,
        'lon_rmse': lon_rmse,
        'alt_rmse': alt_rmse
    }
# %%
errors_optimized = calculate_errors(results_optimized)
errors_original = calculate_errors(results_original)
# %%
def print_errors(errors, label):
    print(f"Errors for {label}:")
    print(f"  Latitude RMSE: {errors['lat_rmse']:.10f}")
    print(f"  Longitude RMSE: {errors['lon_rmse']:.10f}")
    print(f"  Altitude RMSE: {errors['alt_rmse']:.10f}")
# %%
print_errors(errors_optimized, "Optimized VIO")
print_errors(errors_original, "Original VIO")
# %%
# %%
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

# %%
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

    # Построение графиков широты
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(vio_lat_org, label='Original VIO Latitude', linestyle='--')
    plt.plot(vio_lat_opt, label='Optimized VIO Latitude', linestyle='-.')
    plt.plot(gps_lat, label='GPS Latitude', linestyle='-')
    plt.xlabel('Index')
    plt.ylabel('Latitude')
    plt.title('Latitude Comparison')
    plt.legend()
    
    # Построение графиков долготы
    plt.subplot(1, 3, 2)
    plt.plot(vio_lon_org, label='Original VIO Longitude', linestyle='--')
    plt.plot(vio_lon_opt, label='Optimized VIO Longitude', linestyle='-.')
    plt.plot(gps_lon, label='GPS Longitude', linestyle='-')
    plt.xlabel('Index')
    plt.ylabel('Longitude')
    plt.title('Longitude Comparison')
    plt.legend()
    
    # Построение графиков высоты
    plt.subplot(1, 3, 3)
    plt.plot(vio_alt_org, label='Original VIO Altitude', linestyle='--')
    plt.plot(vio_alt_opt, label='Optimized VIO Altitude', linestyle='-.')
    plt.plot(gps_alt, label='GPS Altitude', linestyle='-')
    plt.xlabel('Index')
    plt.ylabel('Altitude (mm)')
    plt.title('Altitude Comparison')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
plot_comparison(results_optimized, results_original)
# %%
# %%
def plot_comparison_transformed(results_optimized, results_original):
    gps_lat = results_original['lat_GPS']
    gps_lon = results_original['lon_GPS']
    gps_alt = results_original['alt_GPS']

    vio_lat_transformed_opt = results_optimized['lat_VIO_transformed']
    vio_lon_transformed_opt = results_optimized['lon_VIO_transformed']
    vio_alt_opt = results_optimized['alt_VIO']

    vio_lat_transformed_org = results_original['lat_VIO_transformed']
    vio_lon_transformed_org = results_original['lon_VIO_transformed']
    vio_alt_org = results_original['alt_VIO']

    plt.figure(figsize=(18, 6))
    # Построение широты
    plt.subplot(1, 3, 1)
    plt.plot(gps_lat, label='GPS Latitude', linestyle='-')
    plt.plot(vio_lat_transformed_org, label='Original VIO Latitude (transformed)', linestyle='--')
    plt.plot(vio_lat_transformed_opt, label='Optimized VIO Latitude (transformed)', linestyle='-.')
    plt.xlabel('Index')
    plt.ylabel('Latitude')
    plt.title('Latitude Comparison (Transformed)')
    plt.legend()

    # Построение долготы
    plt.subplot(1, 3, 2)
    plt.plot(gps_lon, label='GPS Longitude', linestyle='-')
    plt.plot(vio_lon_transformed_org, label='Original VIO Longitude (transformed)', linestyle='--')
    plt.plot(vio_lon_transformed_opt, label='Optimized VIO Longitude (transformed)', linestyle='-.')
    plt.xlabel('Index')
    plt.ylabel('Longitude')
    plt.title('Longitude Comparison (Transformed)')
    plt.legend()

    # Построение высоты
    plt.subplot(1, 3, 3)
    plt.plot(gps_alt, label='GPS Altitude', linestyle='-')
    plt.plot(vio_alt_org, label='Original VIO Altitude', linestyle='--')
    plt.plot(vio_alt_opt, label='Optimized VIO Altitude', linestyle='-.')
    plt.xlabel('Index')
    plt.ylabel('Altitude (mm)')
    plt.title('Altitude Comparison')
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%
plot_comparison_transformed(results_optimized, results_original)
# %%
lat_diff_mean = np.mean(np.array(results_optimized['lat_VIO']) - np.array(results_original['lat_VIO']))
lon_diff_mean = np.mean(np.array(results_optimized['lon_VIO']) - np.array(results_original['lon_VIO']))
gps_lat_diff_mean = np.mean(np.array(results_optimized['lat_VIO']) - np.array(results_original['lat_GPS']))
gps_lon_diff_mean = np.mean(np.array(results_optimized['lon_VIO']) - np.array(results_original['lon_GPS']))

print(f"Mean Latitude Difference (Optimized VIO - Original VIO): {lat_diff_mean:.10f}")
print(f"Mean Longitude Difference (Optimized VIO - Original VIO): {lon_diff_mean:.10f}")
print(f"Mean GPS Latitude Difference (Optimized VIO - Original GPS): {gps_lat_diff_mean:.10f}")
print(f"Mean GPS Longitude Difference (Optimized VIO - Original GPS): {gps_lon_diff_mean:.10f}")