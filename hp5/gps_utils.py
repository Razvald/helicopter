from datetime import datetime, date, timedelta
import numpy as np
import queue

from set_config_gps import config_ublox

from time import time
from serial import Serial
from pyubx2 import UBXReader, NMEA_PROTOCOL

from pymavlink import mavutil


def datetime2text(gps_data):
    if 'time' in gps_data.__dict__.keys():
        if (tic:=gps_data.__dict__['time']) != '':
            gps_data.__dict__['time'] = tic.strftime('%H:%M:%S.%f')
    if 'date' in gps_data.__dict__.keys():
        if (dic:=gps_data.__dict__['date']) != '':
            gps_data.__dict__['date'] = dic.strftime('%d/%m/%Y')

    return gps_data


def calc_GPS_week_time(dic, tic):
    today = datetime.strptime(dic, '%d/%m/%Y').date()
    now = datetime.strptime(tic, '%H:%M:%S.%f').time()
    epoch = date(1980, 1, 6)
    
    epochMonday = epoch - timedelta(epoch.weekday())
    todayMonday = today - timedelta(today.weekday())
    GPS_week = int((todayMonday - epochMonday).days / 7)
    GPS_ms = ((today - todayMonday).days * 24 + now.hour) * 3600000 + now.minute*60000 + now.second*1000 + int(now.microsecond/1000)
    return GPS_week, GPS_ms


def gps2pixhawk(data):
    data_keys = data.keys()
    if not (('GNRMC' in data_keys) and ('GNVTG' in data_keys) and ('GNGGA' in data_keys) and ('GNGSA' in data_keys)):
        return None
    elif data['GNRMC']['status'] == "V":
        return None
        
    FLAGS = mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT
    cogt = data['GNVTG']['cogt']
    sogk = data['GNVTG']['sogk']
    if cogt=='' or sogk=='':
        FLAGS = FLAGS | mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ
        vn = 0
        ve = 0
    else:
        cogt = float(cogt) / 180 * np.pi
        sogm = float(sogk) / 3.6
        vn = sogm * np.cos(cogt)
        ve = sogm * np.sin(cogt)
    GPS_week, GPS_ms = calc_GPS_week_time(data['GNRMC']['date'], data['GNRMC']['time'])

    return     [int(time()*10**6), # Timestamp (micros since boot or Unix epoch)
                0, # GPS sensor id in th, e case of multiple GPS
                FLAGS, # flags to ignore 8, 16, 32 etc
                # (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY) |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY |
                # mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY,
                
                GPS_ms, # GPS time (milliseconds from start of GPS week)
                GPS_week, # GPS week number
                3, # 0-1: no fix, 2: 2D fix, 3: 3D fix. 4: 3D with DGPS. 5: 3D with RTK
                int(data['GNRMC']['lat']*10**7), # Latitude (WGS84), in degrees * 1E7
                int(data['GNRMC']['lon']*10**7), # Longitude (WGS84), in degrees * 1E7
                #data['GNGGA']['alt'], # Altitude (AMSL, not WGS84), in m (positive for up)
                0,
                data['GNGSA']['HDOP'], # GPS HDOP horizontal dilution of precision in m
                data['GNGSA']['VDOP'], # GPS VDOP vertical dilution of precision in m
                float(vn), # GPS velocity in m/s in NORTH direction in earth-fixed NED frame
                float(ve), # GPS velocity in m/s in EAST direction in earth-fixed NED frame
                0, # GPS velocity in m/s in DOWN direction in earth-fixed NED frame
                0.6, # GPS speed accuracy in m/s
                5.0, # GPS horizontal accuracy in m
                3.0, # GPS vertical accuracy in m
                data['GNGGA']['numSV'] # Number of satellites visible,
               ]

class GPSData():
    
    def __init__(self, device="/dev/ttyS0", rate_ms=100, baudrate=115200, timeout=3):
        #self.rate_ms = 100
        #self.baudrate = 115200
        config_ublox(device, rate_ms=rate_ms, baudrate=baudrate)

        self._stream = Serial(device, baudrate=baudrate, timeout=timeout)
        self._ublox_m8n = UBXReader(self._stream, protfilter=NMEA_PROTOCOL)

        self.data = {}

    def run(self, stop_event, outque):
        print('start polling GPS sensor')
        while not stop_event.is_set():
            try:
                # read GPS
                for jj in range(12):
                    _, parsed_data = self._ublox_m8n.read()
                    if parsed_data != None:
                        # convert to string to comply with json dump    
                        gps_data_id = parsed_data.identity
                        self.data[gps_data_id] = datetime2text(parsed_data).__dict__
                    else:
                        self.data = None
                        break
                outque.put(self.data)
                if outque.full():
                    _ = outque.get(timeout=1) 
                
            except Exception as e:
                print(f'GPS sensor error {e}')
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break
                
        self._stream.close()
        print('stop polling GPS sensor')
