import os
import json
from time import time
import matplotlib.pyplot as plt

# %%
# Инициализация параметров
set_dir = '2024_12_15_15_31_8_num_3'

json_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.json')])

start = 0
count_json = len(json_files)

# %%
def run_vio_from_json(json_files, start, count_json, set_dir):
    """
    Считывает данные VIO из JSON-файлов.
    """
    lat_VIO, lon_VIO, alt_VIO = [], [], []

    for filename in json_files[start:start + count_json]:
        file_path = f'{set_dir}/{filename}'
        if os.path.getsize(file_path) == 0:
            print(f"Файл {filename} пустой, пропускаем.")
            continue
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                if 'VIO' in data:
                    lat_VIO.append(data['VIO'].get('lat', 0.0))
                    lon_VIO.append(data['VIO'].get('lon', 0.0))
                    alt_VIO.append(data['VIO'].get('alt', 0.0))
        except json.decoder.JSONDecodeError:
            print(f"Ошибка при чтении файла {filename}, возможно он поврежден или пуст.")
            continue

    return {
        'lat_VIO': lat_VIO,
        'lon_VIO': lon_VIO,
        'alt_VIO': alt_VIO,
    }

# %%
# Основной процесс обработки для первого набора данных
start_time = time()
results = run_vio_from_json(json_files, start, count_json, set_dir)
elapsed_time = time() - start_time

# %%
def transform_vio_coords(vio_lon_list, vio_lat_list):
    """
    Преобразование координат VIO в новые координаты.
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

# Применяем трансформацию для первого набора данных
transformed_lon_org, transformed_lat_org = transform_vio_coords(
    results['lon_VIO'], results['lat_VIO']
)

# Обновляем результаты для построения графиков
results['lon_VIO_transformed'] = transformed_lon_org
results['lat_VIO_transformed'] = transformed_lat_org

# %%
def plot_results_comparison(results):
    """
    Построение графиков на основе данных VIO для двух наборов.
    """
    lat_vio = results['lat_VIO']
    lon_vio = results['lon_VIO']
    
    lat_vio_transf = results['lat_VIO_transformed']
    lon_vio_transf = results['lon_VIO_transformed']
    
    # Построение графиков
    plt.figure(figsize=(12, 6))
    
    # График 1: Траектория VIO с применением трансформации
    plt.subplot(1, 2, 1)
    plt.plot(lon_vio_transf, lat_vio_transf, label='VIO set 1', color='orange', marker='o', markersize=2)
    plt.title('VIO Trajectory (Transformed)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid()
    
    # График 2: Траектория VIO без трансформации
    plt.subplot(1, 2, 2)
    plt.plot(lon_vio, lat_vio, label='VIO set 1', color='orange', marker='o', markersize=2)
    plt.title('VIO Trajectory')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Пример использования:
plot_results_comparison(results)
