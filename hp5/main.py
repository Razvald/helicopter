# main.py

import os
import shutil
from time import sleep, time
import datetime
import traceback
import json
from contextlib import contextmanager

from multiprocessing import Process, Event, Queue

import cv2

from utils import *
from camkey import Camera
from pdata import PosData
from gps_utils import GPSData, gps2pixhawk
import info_on_display

from vio_ort import VIO

from pymavlink import mavutil
from screeninfo import get_monitors
from screeninfo import ScreenInfoError

import cfg  # Import configuration constants


@contextmanager
def NamedWindow(*args, **kwargs):
    name_window = ""
    if "winname" in kwargs:
        name_window = kwargs["winname"]
    else:
        name_window = args[0]
    window = cv2.namedWindow(*args, **kwargs)
    try:
        yield window
    finally:
        cv2.destroyWindow(name_window)


def initialize_mavlink_connection(port):
    master = mavutil.mavlink_connection(port)
    master.wait_heartbeat()
    print(f'Got heartbeat on {port}')
    return master


def initialize_processes(stop, vcap,  posque, gpsque, vidque, pos_queue_window):
    # Initialize position data process
    posdata = PosData()
    data_poll_process = Process(target=posdata.run, args=(stop, posque))
    data_poll_process.start()

    # Initialize GPS data process
    gps = GPSData()
    gps_process = Process(target=gps.run, args=(stop, gpsque))
    gps_process.start()

    
    video_poll_process = Process(target=vcap.run, args=(stop, vidque))
    video_poll_process.start()
    
    #info_on_display_procecc = Process(target=send_info_on_display, args=(stop, pos_queue_window))
    #info_on_display_procecc.start()
    
    return data_poll_process, gps_process, video_poll_process


def get_initial_coordinates(gpsque):
    lat0 = cfg.DEFAULT_LAT
    lon0 = cfg.DEFAULT_LON
    alt0 = cfg.DEFAULT_ALT
    # Wait until we have GPS data
    gps = gpsque.get()
    if gps != None: 
        if gps['GNRMC']['lat'] != '' and gps['GNRMC']['lon'] != '':
        # If GPS is on, set actual coordinates
            lat0 = gps['GNRMC']['lat']
            lon0 = gps['GNRMC']['lon']
            alt0 = gps['GNGGA']['alt']  # TODO: Use altitude from Earth height map
    return lat0, lon0, alt0


def initialize_vio(lat0, lon0, alt0):
    vis_odo = VIO(lat0, lon0, alt0)
    print(f'Starting at coordinates: {lat0}, {lon0}, {alt0}')
    return vis_odo


def get_pos2pixhawk_function(gps_mode, vis_odo):
    if gps_mode == 'gps':
        return gps2pixhawk
    elif gps_mode == 'vio':
        return vis_odo.vio2pixhawk
    else:
        raise ValueError('Invalid GPS mode specified')


def setup_dumping():
    if cfg.DUMP:
        ipynb_dir = os.path.join(cfg.DUMP_DIR, '.ipynb_checkpoints')
        if os.path.isdir(ipynb_dir):
            shutil.rmtree(ipynb_dir)

        total, used, free = get_drive_space(cfg.DUMP_DIR)
        dump_size = get_folder_size(cfg.DUMP_DIR)
        print(f'Size of dump folder is {dump_size}, space left is {free}')
        assert free > cfg.MIN_FREE_SPACE, f'Empty log dir {cfg.DUMP_DIR} to have enough disk space'

        ct = datetime.datetime.now()
        count_files = len(os.listdir(cfg.DUMP_DIR)) + 1
        timestr = f'{ct.year}_{ct.month}_{ct.day}_{ct.hour}_{ct.minute}_{ct.second}_num_{count_files}'
        data_dir = os.path.join(cfg.DUMP_DIR, timestr)
        
        os.makedirs(data_dir)
        return data_dir
    return None


def send_info_on_display(stop_event, posque):
    screen_width = 640
    screen_height = 480
    monitor_exists = False
    
    # объект для нанесения данных pixhawk на изображения
    info_on_image = info_on_display.InfoPixhawkOnDisplay()
    
    # Пробуем три раза подключится к монитору
    for _ in range(3):
        try:
            # Получение размера экрана
            monitor = get_monitors()[0]  # Берем первый монитор
            screen_width = monitor.width
            screen_height = monitor.height
            monitor_exists = True
        except ScreenInfoError:
            sleep(0.5)
        except Exception as err:
            print(err)
            
    with NamedWindow("Info"):
        cv2.setWindowProperty("Info", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while not stop_event.is_set():
            try:
                frame, msg = posque.get()
                if monitor_exists:
                    img = info_on_image.show_info_on_display(frame, msg)
                    img = cv2.resize(img, (screen_width, screen_height), interpolation = cv2.INTER_AREA)
                    cv2.imshow('Info', img)
                    cv2.waitKey(1)
            except KeyboardInterrupt:
                stop_event.set()
            except Exception as err:
                continue


def calibrate_compass(master):
    # start compass calibration for soft iron. Auto reboot on cancel
    master.mav.command_long_send(
                master.target_system,  # target_system
                0, # target_component
                mavutil.mavlink.MAV_CMD_DO_START_MAG_CAL, # command
                0, # confirmation
                0, # p1: mag_mask
                0, # p2: retry
                1, # p3: autosave flag
                0, # p4: delay
                1, # reboot flag
                0, # param6
                0) # param7

    sleep(30)

    # stop compass calibration and reboot 
    master.mav.command_long_send(
                master.target_system,  # target_system
                0, # target_component
                mavutil.mavlink.MAV_CMD_DO_CANCEL_MAG_CAL, # command
                0, # confirmation
                0, # p1: mag_mask
                0, # param2
                0, # param3
                0, # param4
                0, # param5
                0, # param6
                0) # param7

    sleep(5)

def main_loop(master, stop, posque, gpsque, vidque, vis_odo, pos2pixhawk_func, data_dir, pos_queue_window):
    print("main_loop")
    voltage = None
    init = True
    start_imu = 0
    start_frame = 0
    while True:
        
        try:
            tic = time()
            name = str(int(tic * 1000))

            frame, timestemp_frame = vidque.get()
            msg, timestemp_msg = posque.get()
            if init and timestemp_msg != -1:
                start_imu = timestemp_msg
                start_frame = timestemp_frame
                init = False
            # if not pos_queue_window.full():
            #     pos_queue_window.put((frame, msg))
            if not gpsque.empty():
                gps = gpsque.get()
                if gps is not None:
                    for key in gps.keys():
                        msg[key] = gps[key]
            
            msg['VIO'] = vis_odo.add_trace_pt(frame, msg)
            msg = serialize(msg)
            
            
            if cfg.DUMP and data_dir:
                msg_path = os.path.join(data_dir, f'{name}.json')
                img_path = os.path.join(data_dir, f'{name}.jpg')
                cv2.imwrite(img_path, frame)
                with open(msg_path, 'w') as f:
                    json.dump(msg, f)

            gps_data = pos2pixhawk_func(msg)
            master.mav.gps_input_send(*gps_data)

            if 'SYS_STATUS' in msg.keys():
                voltage = msg['SYS_STATUS']['voltage_battery']

            fps = 1 / (time() - tic)
            if timestemp_msg != -1:
                timestemp_frame = abs(timestemp_frame - start_frame) * 1000
                timestemp_msg = abs(timestemp_msg - start_imu)
                print(f'FPS: {fps:.1f}, Voltage: {voltage if voltage else "N/A"}, delta: {abs(timestemp_frame - timestemp_msg)}', end='\r')

        except KeyboardInterrupt:
            # Handle keyboard interrupt
            mode_id = master.mode_mapping()['LAND']

            master.mav.set_mode_send(
                master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            stop.set()

        except Exception as e:
            print(f'Main loop exception: {traceback.format_exc()}')

        if stop.is_set():
            print('\nStopping all processes...')
            break

    # Ensure all processes are properly terminated
    stop.set()


if __name__ == '__main__':
    # os.system(f'xrandr --output HDMI-1 --mode 720x480 --rotate normal')
    # os.system(f'sync')
# Initialize Mavlink connection
    master = initialize_mavlink_connection(cfg.PORT)
    
    sleep(5)
    calibrate_compass(master)

    # Initialize processes and queues
    stop_event = Event()
    pos_queue = Queue(2)
    gps_queue = Queue(2)
    video_queue = Queue(2)
    
    # Очереди для монитора
    pos_queue_window = Queue(2)

    # # Initialize camera
    camera_index = 0
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            sleep(0.25)
            camera_index = idx
            break
    if camera_index is None:
        print('No camera found')
        exit(1)

    vcap = Camera(camera_index)

    processes = initialize_processes(stop_event, vcap, pos_queue, gps_queue, video_queue, pos_queue_window)

    # Get initial GPS coordinates
    latitude, longitude, altitude = get_initial_coordinates(gps_queue)

    # Initialize Visual Inertial Odometry
    visual_odometry = initialize_vio(latitude, longitude, altitude)

    # Get the appropriate function for converting positions to Pixhawk format
    pos_to_pixhawk = get_pos2pixhawk_function(cfg.GPS_MODE, visual_odometry)
    # Setup data dumping if enabled
    data_directory = setup_dumping()
    try:
        # Run the main loop
        main_loop(
            master,
            stop_event,
            pos_queue,
            gps_queue,
            video_queue,
            visual_odometry,
            pos_to_pixhawk,
            data_directory,
            pos_queue_window
        )
    finally:
        # Ensure all processes are joined properly
        for process in processes:
            process.join()
