# %%
import nvtx

with nvtx.annotate("Init: Imports", color="dodgerblue"):
    import matplotlib.pyplot as plt
    import os
    import json
    import cv2

    import vio_ort as vio_ort
# %%
# Инициализация глобальных параметров
with nvtx.annotate("Init: Global VIO & Params", color="dodgerblue"):
    odometry = vio_ort.VIO(lat0=54.889668, lon0=83.1258973333, alt0=0)
    set_dir = '2024_12_15_15_31_8_num_3'
    json_files = sorted([f for f in os.listdir(set_dir) if f.endswith('.json')])
    start = 1000
    count_json = 100
    lat_VIO, lon_VIO, alt_VIO = [], [], []
    lat_GPS, lon_GPS, alt_GPS = [], [], []

# %%
def run_original():
    # Iterate over files in the dataset directory
    for filename in json_files[start:start + count_json]:
        # Read the JSON file
        with open(f'{set_dir}/{filename}', 'r') as file:
            data = json.load(file)
            if 'GNRMC' in data:
                if data['GNRMC']['status'] == 'A':
                    img_path = set_dir + '/' + os.path.splitext(filename)[0] + '.jpg'
                    image = cv2.imread(img_path)

                    result_vio = odometry.add_trace_pt(image, data)

                    lat_VIO.append(result_vio['lat'])
                    lon_VIO.append(result_vio['lon'])
                    alt_VIO.append(data['VIO']['alt'])

                    lat_GPS.append(data['GNRMC'].get('lat', 0.0))
                    lon_GPS.append(data['GNRMC'].get('lon', 0.0))
                    alt_GPS.append(data['GPS_RAW_INT']['alt'])
    return {
        'lat_VIO': lat_VIO,
        'lon_VIO': lon_VIO,
        'lat_GPS': lat_GPS,
        'lon_GPS': lon_GPS
    }
# %%
with nvtx.annotate("Main: Execute", color="black"):
    orig_results = run_original()
# %%
with nvtx.annotate("Math coords", color="black"):
    def draw_graph():
        plt.plot(lon_GPS, lat_GPS, label='GPS', color='blue')
        plt.plot(lon_VIO, lat_VIO, label='VIO', color='red')
        plt.legend()
        plt.show()

# %%
"""with nvtx.annotate("Draw Function", color="darkviolet"):
    draw_graph()"""

# %%
print(len(lat_GPS))
print(len(lat_VIO))