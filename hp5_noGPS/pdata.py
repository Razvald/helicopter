import queue
import time
from pymavlink import mavutil


class PosData():

    def __init__(self, port='udp:127.0.0.1:14551', gps_baudrate=115200, gps_rate_ms=100):
        # set connection with PixHawk
        self._master = mavutil.mavlink_connection(port)
        self._master.wait_heartbeat()
        print(f'got heartbeat on {port}')
        
        # init data storage
        self.pdata = {}
            
    def _fetch_PX(self):
        while msg:=self._master.recv_match(blocking=False):
            text = msg.to_dict()
            type = text['mavpackettype']
            if 'UNKNOWN' not in type:
                self.pdata[type] = text    

    def run(self, stop_event, outque):
        print('start polling pos data')
        
        while not stop_event.is_set():
            try:
                # finally poll pixhawk until empty packet 
                try:
                    self._fetch_PX()
                except Exception as e:
                    print(f'PixHawk read error {e}') #TODO add logger
                
                # отправляем данные на инференс
                if "RAW_IMU" in self.pdata:
                    outque.put((self.pdata, self.pdata["RAW_IMU"]["time_usec"]/1000))
                else:
                    outque.put((self.pdata, -1))
                if outque.full():
                    _ = outque.get(timeout=0.05) 
                    
            except KeyboardInterrupt:
                print('keybopard interrupt in posdata poll process')
                break    
            except queue.Empty:
                continue   
                
        print(f'pixhawk, imu, altimeter polling stopped')


def serialize(data):
    if isinstance(data, dict):
        return {key: serialize(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize(element) for element in data]
    elif isinstance(data, (bool, int, float, str)):
        return data
    else:
        return ''
    