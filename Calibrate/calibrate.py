# %%
import nvtx

with nvtx.annotate("Init: Imports", color="dodgerblue"):
    import matplotlib.pyplot as plt
    import os
    import json
    import cv2
    import numpy as np
    from time import time

    import vio_ort as vio_ort
    import vio_ortO as vio_ort_original
# %%
# Инициализация глобальных параметров
with nvtx.annotate("Init: Global VIO & Params", color="dodgerblue"):
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
                    alt_VIO.append(result_vio['alt'])

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
with nvtx.annotate("Main: Execute opt", color="black"):
    timer = time()
    results_optimized = run_vio(odometry, json_files, start, count_json)
    print(f"Execution time: {time() - timer:.2f} seconds")

with nvtx.annotate("Main: Execute org", color="black"):
    timer = time()
    results_original = run_vio(odometry_org, json_files, start, count_json)
    print(f"Execution time: {time() - timer:.2f} seconds")
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
with nvtx.annotate("Main: Calculate errors", color="black"):
    errors_optimized = calculate_errors(results_optimized)
    errors_original = calculate_errors(results_original)
# %%
def print_errors(errors, label):
    print(f"Errors for {label}:")
    print(f"  Latitude RMSE: {errors['lat_rmse']:.10f}")
    print(f"  Longitude RMSE: {errors['lon_rmse']:.10f}")
    print(f"  Altitude RMSE: {errors['alt_rmse']:.10f}")
# %%
with nvtx.annotate("Print Errors", color="darkgreen"):
    print_errors(errors_optimized, "Optimized VIO")
    print_errors(errors_original, "Original VIO")
# %%
# Функция для построения графика с GPS и VIO
def plot_comparison(results_optimized, results_original):
    plt.figure(figsize=(12, 8))
    
    # Построим график для широты
    plt.subplot(2, 2, 1)
    plt.plot(results_optimized['lon_GPS'], results_optimized['lat_VIO'], label="Optimized VIO", color="red", alpha=0.7)
    plt.plot(results_original['lon_GPS'], results_original['lat_VIO'], label="Original VIO", color="blue", alpha=0.7)
    plt.plot(results_optimized['lon_GPS'], results_optimized['lat_GPS'], label="GPS", color="green", linestyle='--', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Latitude vs Longitude')
    plt.legend()

    # Построим график для высоты
    plt.subplot(2, 2, 2)
    plt.plot(results_optimized['lat_GPS'], results_optimized['alt_VIO'], label="Optimized VIO", color="red", alpha=0.7)
    plt.plot(results_original['lat_GPS'], results_original['alt_VIO'], label="Original VIO", color="blue", alpha=0.7)
    plt.plot(results_optimized['lat_GPS'], results_optimized['alt_GPS'], label="GPS", color="green", linestyle='--', alpha=0.6)
    plt.xlabel('Latitude')
    plt.ylabel('Altitude')
    plt.title('Altitude vs Latitude')
    plt.legend()

    # Построим график для ошибок в координатах
    plt.subplot(2, 2, 3)
    lat_error_opt = np.array(results_optimized['lat_VIO']) - np.array(results_optimized['lat_GPS'])
    lon_error_opt = np.array(results_optimized['lon_VIO']) - np.array(results_optimized['lon_GPS'])
    lat_error_orig = np.array(results_original['lat_VIO']) - np.array(results_original['lat_GPS'])
    lon_error_orig = np.array(results_original['lon_VIO']) - np.array(results_original['lon_GPS'])

    plt.plot(lat_error_opt, lon_error_opt, label="Optimized VIO", color="red", alpha=0.7)
    plt.plot(lat_error_orig, lon_error_orig, label="Original VIO", color="blue", alpha=0.7)
    plt.xlabel('Latitude Error')
    plt.ylabel('Longitude Error')
    plt.title('Error in Latitude and Longitude')
    plt.legend()

    # Построим график для ошибок по высоте
    plt.subplot(2, 2, 4)
    alt_error_opt = np.array(results_optimized['alt_VIO']) - np.array(results_optimized['alt_GPS'])
    alt_error_orig = np.array(results_original['alt_VIO']) - np.array(results_original['alt_GPS'])

    plt.plot(results_optimized['lat_GPS'], alt_error_opt, label="Optimized VIO", color="red", alpha=0.7)
    plt.plot(results_original['lat_GPS'], alt_error_orig, label="Original VIO", color="blue", alpha=0.7)
    plt.xlabel('Latitude')
    plt.ylabel('Altitude Error')
    plt.title('Error in Altitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

# %%
with nvtx.annotate("Plot Comparison", color="purple"):
    plot_comparison(results_optimized, results_original)

# %%
print(len(results_optimized))
print(len(results_original))