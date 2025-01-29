import cv2
import queue
import time
import logger_copter

logging = logger_copter.CopterLogger(__name__, "/home/orangepi/Log")

class Camera():

    def __init__(self, cap_id):
        self._id = cap_id
        self._cap = cv2.VideoCapture(cap_id)
        if not self._cap.isOpened():
            logging.error(f'can not connect to camera with id {cap_id}')
        assert self._cap.isOpened(), f'can not connect to camera with id {cap_id}'
        # Задаем целевое разрешение
        target_width, target_height = 640, 480
        # FPS
        fps_cam = 30
        # Устанавливаем формат MJPG для камеры
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Устанавливаем разрешение для камеры
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        # Устанавливаем желаемый FPS
        self._cap.set(cv2.CAP_PROP_FPS, fps_cam)
        
    def run(self, stop_event, outque):
        print('start polling frames')
        # Разрешение для inference
        while not stop_event.is_set():
            try:
                ret, frame = self._cap.read()
                timestemp = time.time()
                if not ret:
                    logging.error("Not frame!")
                    continue
                outque.put((frame, timestemp))
                if outque.full():
                    _ = outque.get(timeout=0.1)
                        
            except KeyboardInterrupt:
                print('keybopard interrupt in camera process')
                break
            except queue.Empty:
                continue
            except queue.Full:
                continue
        print(f'frames polling stopped   {not self._cap.isOpened()}',)
            
    def __del__(self):                   
        self._cap.release()
        
        