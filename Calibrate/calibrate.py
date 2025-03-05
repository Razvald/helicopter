# %%
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
from time import time

import vio_ort_org_copy as vio_ort
import vio_ort_org as vio_ort_original
# %%
# Инициализация глобальных параметров
odometry = vio_ort.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
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
results_optimized = run_vio(odometry, json_files, start, count_json)
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
# Функция для построения графика с GPS и VIO
def plot_comparison(results_optimized, results_original):
    vio_lat_org = results_original['lat_VIO']
    vio_lon_org = results_original['lon_VIO']
    vio_alt_org = results_original['alt_VIO']
    
    vio_lat_opt = results_optimized['lat_VIO']
    vio_lon_opt = results_optimized['lon_VIO']
    vio_alt_opt = results_optimized['alt_VIO']
    
    plt.figure(figsize=(18, 6))
    
    

    plt.tight_layout()
    plt.show()

# %%
plot_comparison(results_optimized, results_original)

# %%
lat_diff_mean = np.mean(np.array(results_optimized['lat_VIO']) - np.array(results_original['lat_VIO']))
lon_diff_mean = np.mean(np.array(results_optimized['lon_VIO']) - np.array(results_original['lon_VIO']))

print(f"Mean Latitude Difference (Optimized VIO - Original VIO): {lat_diff_mean:.10f}")
print(f"Mean Longitude Difference (Optimized VIO - Original VIO): {lon_diff_mean:.10f}")