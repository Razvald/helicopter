# %%
import nvtx
import vio_ort
import matplotlib.pyplot as plt
import os
import json
import cv2
import concurrent.futures
import threading
from collections import defaultdict
import plotly.graph_objects as go

# %%
odometry = vio_ort.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)

# Путь к папке
set_dir = '2024_12_15_15_31_8_num_3'

# Получение всех файлов с расширением .json
json_files = [f for f in os.listdir(set_dir) if f.endswith('.json')]

# Сортировка файлов по имени
json_files.sort()

start = 0
count_json = len(json_files)

lat_VIO = []
lon_VIO = []

lat_GPS = []
lon_GPS = []

alt_VIO = []
alt_GPS = []

# %%
# Инициализация структуры для ошибок
fails_collect = defaultdict(lambda: {'num': 0, 'files': []})

# %%
lock = threading.Lock()

# %%
@nvtx.annotate("Register Error", color="red")
def register_error(error_type, filename):
    with lock:
        if error_type not in fails_collect:
            fails_collect[error_type] = {'num': 0, 'files': []}
        fails_collect[error_type]['num'] += 1
        fails_collect[error_type]['files'].append(filename)

# %%
@nvtx.annotate("Process File", color="blue")
def process_file(filename):
    with nvtx.annotate("Load JSON", color="cyan"):
        try:
            with open(os.path.join(set_dir, filename), 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            register_error("JSON decode error", filename)
            return
        except Exception:
            register_error("Processing error", filename)
            return

    with nvtx.annotate("Validate Data", color="magenta"):
        if 'GNRMC' not in data or 'VIO' not in data:
            register_error("Missing GNRMC or VIO", filename)
            return

        if data['GNRMC'].get('status') != 'A':
            register_error("GNRMC status not 'A'", filename)
            return

    with nvtx.annotate("Load Image", color="yellow"):
        img_path = os.path.join(set_dir, os.path.splitext(filename)[0] + '.jpg')
        if not os.path.exists(img_path):
            register_error("Image not found", filename)
            return

        image = cv2.imread(img_path)
        if image is None:
            register_error("Failed to load image", filename)
            return
        
    with nvtx.annotate("Process VIO", color="green"):
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

    with nvtx.annotate("Process GPS", color="orange"):
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
  with nvtx.annotate("ThreadPool Execution", color="purple"):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
      executor.map(process_file, json_files[start:start + count_json])
  
  # Вывод отчетности
  with nvtx.annotate("Report Errors", color="pink"):
    print("\nError Report:")
    for error_type, error_info in fails_collect.items():
      print(f"{error_type} - {error_info['num']} occurrences")
      print(f"Files: {', '.join(error_info['files'])}")
      print()
  
  with nvtx.annotate("Write Debug Points", color="gray"):
    with open("Debugs/debug_points.txt", "w") as f:
      f.write("")
      for i in range(len(lat_GPS)):
        f.write(f'Point num {i}\n')
        f.write(f'{i} point GPS lat: {lat_GPS[i]}\n')
        f.write(f'{i} point VIO lat: {lat_VIO[i]}\n')
        f.write(f'{i} point GPS lon: {lon_GPS[i]}\n')
        f.write(f'{i} point VIO lon: {lon_VIO[i]}\n')

# %%
with nvtx.annotate("Main Function", color="darkviolet"):
    main()

# %%
print(len(lat_GPS))
print(len(lat_VIO))

# %%
alt_VIO = [ x * 1000 for x in alt_VIO]

# %%
# Создание 3D графика
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Построение маршрута
ax.plot(lat_VIO, lon_VIO, alt_VIO, label='Маршрут VIO', color='blue')
ax.plot(lat_GPS, lon_GPS, alt_GPS, label='Маршрут GPS', color='red')

# Настройки графика
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Altitude')
ax.legend()

# Отображение графика
plt.show()


# %%
draw_cinema = False

# %%
if draw_cinema:
    # Пример данных (замените на ваши данные)
    lat = lat_GPS
    lon = lon_GPS
    alt = alt_GPS

    """lat = lat_VIO
    lon = lon_VIO
    alt = alt_VIO"""

    step = 1

    # Создание фигуры с анимацией
    fig = go.Figure()

    # Добавление начального состояния
    fig.add_trace(go.Scatter3d(
        x=[lat[0]],
        y=[lon[0]],
        z=[alt[0]],
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color=alt[0], colorscale='Viridis')
    ))

    # Создание кадров анимации
    frames = []
    for i in range(1, len(lat)):
        frame = go.Frame(
            data=[go.Scatter3d(
                x=lat[:i+step],  # Данные до текущего кадра
                y=lon[:i+step],
                z=alt[:i+step],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4, color=alt[:i+step], colorscale='Viridis')
            )]
        )
        frames.append(frame)

    fig.frames = frames

    # Добавление кнопок управления анимацией
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Старт',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Стоп',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )

    # Настройка размеров и осей
    fig.update_layout(
        scene=dict(
            xaxis_title='Широта',
            yaxis_title='Долгота',
            zaxis_title='Высота',
        ),
        title='Анимация маршрута дрона',
        width=1200,
        height=800,
    )

    fig.show()

# %%
# Списки для данных
gps_lats = lat_GPS.copy()
gps_lons = lon_GPS.copy()
vio_lats = lat_VIO.copy()
vio_lons = lon_VIO.copy()

gps_lats = [ -x + 2 * 54.8894116667 for x in gps_lats]

#vio_lats = [ -x + 2 * 54.8894116667 for x in vio_lats]
vio_lons = [ -x + 2 * 83.1258973333 for x in vio_lons]

# Визуализация данных с аномалиями
plt.figure(figsize=(6, 5))

plt.plot(gps_lons, gps_lats, color="black", label="GPS")
plt.plot(vio_lons, vio_lats, color="red", label="VIO")

plt.title("Сравнение траекторий GPS и VIO")
plt.xlabel("Долгота")
plt.ylabel("Широта")
plt.legend()
plt.grid()
plt.show()

# %%
copy_lat_VIO = lat_VIO.copy()
copy_lon_VIO = lon_VIO.copy()

copy_lat_GPS = lat_GPS.copy()
copy_lon_GPS = lon_GPS.copy()

# Создаем фигуру и оси
plt.figure(figsize=(10, 5))

# Преобразование для глобальных координат широты
copy_lat_GPS = [ -x + 2 * 54.8894116667 for x in copy_lat_GPS]

# Рисуем первый график (GPS)
plt.plot(copy_lon_GPS, copy_lat_GPS, label='GPS', color='blue')

# Рисуем второй график (VIO)
plt.plot(copy_lon_VIO, copy_lat_VIO, label='VIO', color='red')

# Добавляем заголовок и метки осей
plt.title('Графики GPS и VIO (с глобальными координатами)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Добавляем легенду
plt.legend()

# Отображение графика
plt.tight_layout()
plt.show()


