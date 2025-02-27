# %%
import nvtx
import matplotlib.pyplot as plt
import os
import json
import cv2
import concurrent.futures
import threading
from collections import defaultdict
# %%
import Calibrate.vio_ort as vio_ort

# %%
# Инициализация глобальных параметров
with nvtx.annotate("Init: Global VIO & Params", color="dodgerblue"):
    odometry = vio_ort.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
    set_dir = '2024_12_15_15_31_8_num_3'
    json_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.json')])
    start = 1000
    count_json = 1000
    lat_VIO, lon_VIO = [], []
    lat_GPS, lon_GPS = [], []
    alt_VIO, alt_GPS = [], []

# %%
with nvtx.annotate("Init: Error Collection", color="red"):
        fails_collect = defaultdict(lambda: {'num': 0, 'files': []})
        lock = threading.Lock()

# %%
@nvtx.annotate("Error: Register", color="red")
def register_error(error_type, filename):
        with lock:
                fails_collect[error_type]['num'] += 1
                fails_collect[error_type]['files'].append(filename)

# %%
@nvtx.annotate("File: Processing", color="blue")
def process_file(filename):
    with nvtx.annotate("JSON: Load", color="cyan"):
        try:
            with open(os.path.join(set_dir, filename), 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            register_error("JSON decode error", filename)
            return
        except Exception:
            register_error("Processing error", filename)
            return

    with nvtx.annotate("Data: Validate", color="magenta"):
        if 'GNRMC' not in data or 'VIO' not in data:
            register_error("Missing GNRMC or VIO", filename)
            return
        if data['GNRMC'].get('status') != 'A':
            register_error("GNRMC status not 'A'", filename)
            return

    with nvtx.annotate("Image: Load", color="yellow"):
        img_path = os.path.join(set_dir, os.path.splitext(filename)[0] + '.jpg')
        if not os.path.exists(img_path):
            register_error("Image not found", filename)
            return
        image = cv2.imread(img_path)
        if image is None:
            register_error("Failed to load image", filename)
            return

    with nvtx.annotate("VIO: Process", color="green"):
        try:
            result_vio = odometry.add_trace_pt(image, data)
            if 'lat' not in result_vio or 'lon' not in result_vio:
                register_error("VIO result missing 'lat' or 'lon'", filename)
                return
            with lock:
                lat_VIO.append(result_vio['lat'])
                lon_VIO.append(result_vio['lon'])
                alt_VIO.append(data['VIO']['alt'])
        except Exception as e:
            register_error("VIO processing error", filename)
            return

    with nvtx.annotate("GPS: Process", color="orange"):
        try:
            with lock:
                lat_GPS.append(data['GNRMC'].get('lat', 0.0))
                lon_GPS.append(data['GNRMC'].get('lon', 0.0))
                alt_GPS.append(data['GPS_RAW_INT']['alt'])
        except KeyError:
            register_error("GPS data missing", filename)

# %% [markdown]
# Путем перебора наибольшая выгода при 6 потоках

# %%
def main():
    workers = 6
    with nvtx.annotate("ThreadPool: Execute", color="purple"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(process_file, json_files[start:start + count_json])

    # Вывод отчетности
    with nvtx.annotate("Report Errors", color="pink"):
        print("\nError Report:")
        for error_type, error_info in fails_collect.items():
            print(f"{error_type} - {error_info['num']} occurrences")
            print(f"Files: {', '.join(error_info['files'])}")
            print()

# %%
with nvtx.annotate("Main: Execute", color="darkviolet"):
    main()

# %%
with nvtx.annotate("Math coords", color="black"):
    def draw_graph():
        # Шаг 1. Загрузка координат
        gps_lat = lat_GPS.copy()
        gps_lon = lon_GPS.copy()
        vio_lat = lat_VIO.copy()
        vio_lon = lon_VIO.copy()
        gps_alt = alt_GPS.copy()
        vio_alt = alt_VIO.copy()

        # Шаг 2. Сохраняем начальные точки (они должны совпадать)
        gps_lon0 = gps_lon[0]
        gps_lat0 = gps_lat[0]
        vio_lon0 = vio_lon[0]
        vio_lat0 = vio_lat[0]

        # Шаг 3. Вычисляем средние изменения (дельты) для последовательностей координат
        mean_gps_lon_diff = sum(abs(gps_lon[i + 1] - gps_lon[i]) for i in range(len(gps_lon) - 1)) / (len(gps_lon) - 1)
        mean_gps_lat_diff = sum(abs(gps_lat[i + 1] - gps_lat[i]) for i in range(len(gps_lat) - 1)) / (len(gps_lat) - 1)

        mean_vio_lon_diff = sum(abs(vio_lon[i + 1] - vio_lon[i]) for i in range(len(vio_lon) - 1)) / (len(vio_lon) - 1)
        mean_vio_lat_diff = sum(abs(vio_lat[i + 1] - vio_lat[i]) for i in range(len(vio_lat) - 1)) / (len(vio_lat) - 1)

        # Шаг 4. Вычисляем масштабные коэффициенты
        # Здесь предположено, что оси VIO перепутаны:
        # - GPS долгота (горизонталь) соответствует VIO "широте" (vio_lat)
        # - GPS широта (вертикаль) соответствует VIO "долготе" (vio_lon)
        scale_for_lon = mean_gps_lon_diff / mean_vio_lat_diff  # Для преобразования VIO широты -> GPS долгота
        scale_for_lat = mean_gps_lat_diff / mean_vio_lon_diff  # Для преобразования VIO долготы -> GPS широта

        # Шаг 5. Сохраняем параметры трансформации в JSON
        transformation_params = {
                "gps_lon0": gps_lon0,
                "gps_lat0": gps_lat0,
                "vio_lon0": vio_lon0,
                "vio_lat0": vio_lat0,
                "scale_for_lon": scale_for_lon,
                "scale_for_lat": scale_for_lat
        }

        with open("Debugs/transformation_params.json", "w") as f:
            json.dump(transformation_params, f, indent=4)

        # Шаг 6. Определяем функцию для преобразования VIO координат с использованием сохранённых параметров
        def transform_vio_coords(vio_lon_list, vio_lat_list, params):
            """
            Преобразование координат VIO по сохранённым параметрам.
            Аргументы:
                vio_lon_list: список VIO долготы (будет использоваться для расчёта GPS широты)
                vio_lat_list: список VIO широты (будет использоваться для расчёта GPS долготы)
                params: словарь с параметрами трансформации
            Возвращает:
                transformed_lon: список преобразованных GPS долготы
                transformed_lat: список преобразованных GPS широты
                """
            gps_lon0 = params["gps_lon0"]
            gps_lat0 = params["gps_lat0"]
            vio_lon0 = params["vio_lon0"]
            vio_lat0 = params["vio_lat0"]
            scale_for_lon = params["scale_for_lon"]
            scale_for_lat = params["scale_for_lat"]

            # Преобразование:
            # Для GPS долготы используем VIO широту, сдвигаем и масштабируем:
            transformed_lon = [(v_lat - vio_lat0) * scale_for_lon + gps_lon0 for v_lat in vio_lat_list]
            # Для GPS широты используем VIO долготу, но с инверсией (так как ось перевёрнута):
            transformed_lat = [-(v_lon - vio_lon0) * scale_for_lat + gps_lat0 for v_lon in vio_lon_list]
            return transformed_lon, transformed_lat

        # Применяем трансформацию к имеющимся данным (для демонстрации)
        vio_lon_transformed, vio_lat_transformed = transform_vio_coords(vio_lon, vio_lat, transformation_params)

        # Преобразуем высоту VIO в метры (делим на 10, так как в VIO высота в дециметрах)
        vio_alt_meters = [v_alt * 1000 for v_alt in vio_alt]

        with nvtx.annotate("3D Graph", color="red"):
            # Создаем несколько графиков с разными углами обзора
            fig = plt.figure(figsize=(18, 14))

            # Первый график — угол 30 по вертикали и 60 по горизонтали
            ax1 = fig.add_subplot(231, projection='3d')
            ax1.plot(gps_lon, gps_lat, gps_alt, linestyle="-", color="blue", label="GPS")
            ax1.plot(vio_lon_transformed, vio_lat_transformed, vio_alt_meters, linestyle="--", color="red", label="VIO (трансформированные)")
            ax1.set_xlabel('Долгота', fontsize=10)  # Уменьшаем размер шрифта
            ax1.set_ylabel('Широта', fontsize=10)
            ax1.set_title('Вид 1: 90° по вертикали, -90° по горизонтали', fontsize=12)
            ax1.view_init(elev=90, azim=-90)
            ax1.legend()
            ax1.tick_params(axis='both', which='major', labelsize=8)  # Уменьшаем размер меток осей

            # Второй график — угол 45 по вертикали и 90 по горизонтали
            ax2 = fig.add_subplot(232, projection='3d')
            ax2.plot(gps_lon, gps_lat, gps_alt, linestyle="-", color="blue", label="GPS")
            ax2.plot(vio_lon_transformed, vio_lat_transformed, vio_alt_meters, linestyle="--", color="red", label="VIO (трансформированные)")
            ax2.set_xlabel('Долгота', fontsize=10)
            ax2.set_zlabel('Высота (метры)', fontsize=10)
            ax2.set_title('Вид 2: 0° по вертикали, 90° по горизонтали', fontsize=12)
            ax2.view_init(elev=0, azim=-90)
            ax2.legend()
            ax2.tick_params(axis='both', which='major', labelsize=8)

            # Третий график — угол 60 по вертикали и 180 по горизонтали
            ax3 = fig.add_subplot(233, projection='3d')
            ax3.plot(gps_lon, gps_lat, gps_alt, linestyle="-", color="blue", label="GPS")
            ax3.plot(vio_lon_transformed, vio_lat_transformed, vio_alt_meters, linestyle="--", color="red", label="VIO (трансформированные)")
            ax3.set_xlabel('Долгота', fontsize=10)
            ax3.set_ylabel('Широта', fontsize=10)
            ax3.set_zlabel('Высота (метры)', fontsize=10)
            ax3.set_title('Вид 3: 60° по вертикали, 180° по горизонтали', fontsize=12)
            ax3.view_init(elev=60, azim=180)
            ax3.legend()
            ax3.tick_params(axis='both', which='major', labelsize=8)

            # Автоматически подгоняем графики по размеру
            plt.tight_layout()

            # Скрываем метки оси
            ax1.set_zticks([])
            ax1.set_zticklabels([])

            # Скрываем метки оси
            ax2.set_yticks([])
            ax2.set_yticklabels([])

            # Показать графики
            plt.show()

# %%
with nvtx.annotate("Draw Function", color="darkviolet"):
    draw_graph()

# %%
with nvtx.annotate("Math coords", color="black"):
    def draw_graph2():
        gps_lat = lat_GPS.copy()
        gps_lon = lon_GPS.copy()
        vio_lat = lat_VIO.copy()
        vio_lon = lon_VIO.copy()
        gps_alt = alt_GPS.copy()
        vio_alt = alt_VIO.copy()

        gps_lon0 = gps_lon[0]
        gps_lat0 = gps_lat[0]
        vio_lon0 = vio_lon[0]
        vio_lat0 = vio_lat[0]

        gps_lon_range = max(gps_lon) - min(gps_lon)
        gps_lat_range = max(gps_lat) - min(gps_lat)
        vio_lon_range = max(vio_lon) - min(vio_lon)
        vio_lat_range = max(vio_lat) - min(vio_lat)

        scale_for_lon = gps_lon_range / vio_lat_range  # VIO широта -> GPS долгота
        scale_for_lat = gps_lat_range / vio_lon_range  # VIO долгота -> GPS широта

        transformation_params = {
            "gps_lon0": gps_lon0,
            "gps_lat0": gps_lat0,
            "vio_lon0": vio_lon0,
            "vio_lat0": vio_lat0,
            "scale_for_lon": scale_for_lon,
            "scale_for_lat": scale_for_lat
        }

        with open("Debugs/transformation_params.json", "w") as f:
            json.dump(transformation_params, f, indent=4)

        def transform_vio_coords(vio_lon_list, vio_lat_list, params):
            gps_lon0 = params["gps_lon0"]
            gps_lat0 = params["gps_lat0"]
            vio_lon0 = params["vio_lon0"]
            vio_lat0 = params["vio_lat0"]
            scale_for_lon = params["scale_for_lon"]
            scale_for_lat = params["scale_for_lat"]

            transformed_lon = [(v_lat - vio_lat0) * scale_for_lon + gps_lon0 for v_lat in vio_lat_list]
            transformed_lat = [-(v_lon - vio_lon0) * scale_for_lat + gps_lat0 for v_lon in vio_lon_list]
            return transformed_lon, transformed_lat

        vio_lon_transformed, vio_lat_transformed = transform_vio_coords(vio_lon, vio_lat, transformation_params)
        vio_alt_meters = [v_alt * 1000 for v_alt in vio_alt]

        @nvtx.annotate("3D Graph", color="red")
        def plot_draw(vio_lat):
            fig = plt.figure(figsize=(18, 14))

            # Вид 1: 90° по вертикали, -90° по горизонтали
            ax1 = fig.add_subplot(231, projection='3d')
            ax1.plot(gps_lon, gps_lat, gps_alt, linestyle="-", color="blue", label="GPS")
            ax1.plot(vio_lon_transformed, vio_lat, vio_alt_meters, linestyle="--", color="red", label="VIO (трансформированные)")
            ax1.set_xlabel('Долгота', fontsize=10)
            ax1.set_ylabel('Широта', fontsize=10)
            ax1.set_title('Вид 1: 90° по вертикали, -90° по горизонтали', fontsize=12)
            ax1.view_init(elev=90, azim=-90)
            ax1.legend()
            ax1.tick_params(axis='both', which='major', labelsize=8)

            # Вид 2: 0° по вертикали, 90° по горизонтали
            ax2 = fig.add_subplot(232, projection='3d')
            ax2.plot(gps_lon, gps_lat, gps_alt, linestyle="-", color="blue", label="GPS")
            ax2.plot(vio_lon_transformed, vio_lat, vio_alt_meters, linestyle="--", color="red", label="VIO (трансформированные)")
            ax2.set_xlabel('Долгота', fontsize=10)
            ax2.set_zlabel('Высота (метры)', fontsize=10)
            ax2.set_title('Вид 2: 0° по вертикали, 90° по горизонтали', fontsize=12)
            ax2.view_init(elev=0, azim=-90)
            ax2.legend()
            ax2.tick_params(axis='both', which='major', labelsize=8)

            # Вид 3: 60° по вертикали, 180° по горизонтали
            ax3 = fig.add_subplot(233, projection='3d')
            ax3.plot(gps_lon, gps_lat, gps_alt, linestyle="-", color="blue", label="GPS")
            ax3.plot(vio_lon_transformed, vio_lat, vio_alt_meters, linestyle="--", color="red", label="VIO (трансформированные)")
            ax3.set_xlabel('Долгота', fontsize=10)
            ax3.set_ylabel('Широта', fontsize=10)
            ax3.set_zlabel('Высота (метры)', fontsize=10)
            ax3.set_title('Вид 3: 60° по вертикали, 180° по горизонтали', fontsize=12)
            ax3.view_init(elev=60, azim=180)
            ax3.legend()
            ax3.tick_params(axis='both', which='major', labelsize=8)

            plt.tight_layout()
            plt.show()

        plot_draw(vio_lat_transformed)


        delta_lat = [gps_lat[i] - vio_lat_transformed[i] for i in range(len(gps_lat))]
        plt.plot(delta_lat, label="Разница (GPS - VIO широта)")
        plt.xlabel("Индекс точки")
        plt.ylabel("Разница в широте")
        plt.legend()
        plt.title("Погрешность VIO по широте")
        plt.grid()
        plt.show()

        """for i in range(0, len(delta_lat), 50):
            print(f"delta_lat[{i}]: {delta_lat[i]:.10f}")"""

        def plot_comparison(gps_lat, vio_lat):
            plt.figure(figsize=(10, 6))
            plt.plot(gps_lat, label="GPS Широта", color="blue")
            plt.plot(vio_lat, label="VIO Широта (скорректированная)", color="red", linestyle="--")
            plt.xlabel("Индекс точки")
            plt.ylabel("Широта")
            plt.legend()
            plt.title("Сравнение GPS и VIO Широты")
            plt.grid(True)
            plt.show()

        plot_comparison(gps_lat, vio_lat_transformed)

# %%
with nvtx.annotate("Draw Function", color="darkviolet"):
    draw_graph2()

# %%
print(len(lat_GPS))
print(len(lat_VIO))