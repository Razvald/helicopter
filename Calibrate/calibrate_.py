# %%
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
from time import time

import vio_ort_org as vio_ort_original
# %%
# Инициализация глобальных параметров
odometry_org = vio_ort_original.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
set_dir = '2024_12_15_15_31_8_num_3'
json_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.json')])
start = 1000
count_json = 100
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
errors_original = calculate_errors(results_original)
# %%
def print_errors(errors, label):
    print(f"Errors for {label}:")
    print(f"  Latitude RMSE: {errors['lat_rmse']:.10f}")
    print(f"  Longitude RMSE: {errors['lon_rmse']:.10f}")
    print(f"  Altitude RMSE: {errors['alt_rmse']:.10f}")
# %%
print_errors(errors_original, "Original VIO")
# %%
# Загрузка параметров трансформации для оптимизированного варианта
with open("Debugs/transformation_params.json", "r") as f:
    transformation_params = json.load(f)

def transform_vio_coords(vio_lon_list, vio_lat_list, params):
    gps_lon0 = params["gps_lon0"]
    gps_lat0 = params["gps_lat0"]
    vio_lon0 = params["vio_lon0"]
    vio_lat0 = params["vio_lat0"]
    scale_for_lon = params["scale_for_lon"]
    scale_for_lat = params["scale_for_lat"]

    # Для GPS долготы используем VIO широту, для GPS широты — VIO долготу (с инверсией)
    transformed_lon = [(v_lat - vio_lat0) * scale_for_lon + gps_lon0 for v_lat in vio_lat_list]
    transformed_lat = [-(v_lon - vio_lon0) * scale_for_lat + gps_lat0 for v_lon in vio_lon_list]
    return transformed_lon, transformed_lat

# Применяем трансформацию к данным оптимизированного варианта
opt_vio_transformed_lon, opt_vio_transformed_lat = transform_vio_coords(
    results_original['lon_VIO'],  # передаём исходные VIO долготы
    results_original['lat_VIO'],  # передаём исходные VIO широты
    transformation_params
)

# Обновляем результаты для построения графика
results_original['lon_VIO'] = opt_vio_transformed_lon
results_original['lat_VIO'] = opt_vio_transformed_lat
# %%
# Функция для построения графика с GPS и VIO
def plot_comparison(results_original):
    gps_lat = results_original['lat_GPS']
    gps_lon = results_original['lon_GPS']
    gps_alt = results_original['alt_GPS']
    
    vio_lat = results_original['lat_VIO']
    vio_lon = results_original['lon_VIO']
    vio_alt = results_original['alt_VIO']
    
    plt.figure(figsize=(18, 6))
    
    # Построим график для широты
    plt.subplot(1, 3, 1)
    plt.plot(vio_lon, vio_lat, label="Original VIO", color="blue", alpha=0.7)
    plt.plot(gps_lon, gps_lat, label="Original GPS", color="red", alpha=0.7)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Latitude vs Longitude')
    plt.legend()

    # Построим график для высоты
    plt.subplot(1, 3, 2)
    plt.plot(vio_lat, vio_alt, label="Original VIO", color="blue", alpha=0.7)
    plt.plot(gps_lat, gps_alt, label="Original GPS", color="red", alpha=0.7)
    plt.xlabel('Latitude')
    plt.ylabel('Altitude')
    plt.title('Altitude vs Latitude')
    plt.legend()

    # Построим график для ошибок в координатах
    plt.subplot(1, 3, 3)
    lat_error_orig = np.array(vio_lat) - np.array(gps_lat)
    lon_error_orig = np.array(vio_lon) - np.array(gps_lon)

    # Используем scatter для отрисовки точек с градиентом
    points = plt.scatter(lat_error_orig, lon_error_orig, c=np.arange(len(lat_error_orig)), cmap='viridis', alpha=0.7)
    plt.xlabel('Latitude Error')
    plt.ylabel('Longitude Error')
    plt.title('Error in Latitude and Longitude')
    plt.colorbar(points, label='Index')
    plt.legend(["Original VIO"])

    plt.tight_layout()
    plt.show()

# %%
#plot_comparison(results_original)