import math

import numpy as np
import cv2


class InfoPixhawkOnDisplay():
    def __init__(self):
        self.local_distance = 0
        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.battery = 0
        self.height = 0
        self.fly_mode = 0
        self.count_mode = 0
        self.max_count = 60
    
    # Цвет в формате BGR (например, белый)
    def print_text_to_img(self, img, text, x, y, text_color = (255, 255, 255), font_scale=0.9):
        text_position = (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        # Получите размер текста
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )

        # Координаты для фона
        x, y = text_position
        background_top_left = (x, y - text_height - baseline)
        background_bottom_right = (x + text_width, y + baseline)
        background_color = (0, 0, 0)  # Цвет фона в формате BGR (черный)

        # Нарисуйте прямоугольник фона
        cv2.rectangle(
            img, background_top_left, background_bottom_right, background_color, -1
        )
        # Добавьте текст поверх фона
        cv2.putText(
            img,
            text,
            text_position,
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return img

    # Отрисовка компаса
    def compas_print(self, img, angle_radians):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Задайте параметры компаса
        compass_center = (80, 120)  # Позиция центра компаса (левый верхний угол)
        compass_radius = 40  # Радиус компаса
        arrow_length = compass_radius - 5  # Длина стрелки
        line_thickness = 2  # Толщина линий
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.7  # Размер текста
        text_thickness_outer = 3  # Толщина для черной обводки текста
        text_thickness_inner = 1  # Толщина для белого текста
        text_color_outer = (0, 0, 0)  # Цвет обводки текста (черный)
        text_color_inner = (255, 255, 255)  # Цвет внутреннего текста (белый)

        # Находим конечные координаты стрелки
        black_arrow_end = (
            int(compass_center[0] + arrow_length * math.sin(angle_radians)),
            int(compass_center[1] - arrow_length * math.cos(angle_radians)),
        )
        white_arrow_end = (
            int(compass_center[0] + (arrow_length - 5) * math.sin(angle_radians)),
            int(compass_center[1] - (arrow_length - 5) * math.cos(angle_radians)),
        )

        # Нарисуйте круг для компаса
        cv2.circle(img, compass_center, compass_radius, text_color_inner, line_thickness)

        # Нарисуйте черную часть стрелки (более длинная)
        cv2.arrowedLine(
            img,
            compass_center,
            black_arrow_end,
            (0, 0, 0),
            line_thickness + 1,
            tipLength=0.3,
        )

        # Нарисуйте белую часть стрелки (короче, поверх черной)
        cv2.arrowedLine(
            img,
            compass_center,
            white_arrow_end,
            (255, 255, 255),
            line_thickness,
            tipLength=0.3,
        )

        # Добавьте текст для направлений
        offset = 50 # Отступ текста от центра компаса
        directions = {
            "N": (compass_center[0] - 6, compass_center[1] - offset),
            "W": (compass_center[0] - offset - 10, compass_center[1] + 8),
            "E": (compass_center[0] + offset - 3, compass_center[1] + 8),
        }
        for direction, position in directions.items():
            # Нарисуйте черный текст как обводку
            cv2.putText(
                img,
                direction,
                position,
                text_font,
                text_scale,
                text_color_outer,
                text_thickness_outer,
                cv2.LINE_AA,
            )
            # Нарисуйте белый текст поверх черного
            cv2.putText(
                img,
                direction,
                position,
                text_font,
                text_scale,
                text_color_inner,
                text_thickness_inner,
                cv2.LINE_AA,
            )

        # Печатаем угол в градусах
        angle_degree = int(math.degrees(angle_radians)) % 360
        compas_text = f"{angle_degree}"
        (text_width, _), _ = cv2.getTextSize(compas_text, font, font_scale, font_thickness)
        angles_position = (
            compass_center[0] - text_width // 2,
            compass_center[1] + offset + 30,
        )

        img = self.print_text_to_img(img, compas_text, angles_position[0], angles_position[1])

        return img

    # Отрисовка батареи
    def battery_print(self, img, charge_battery):
        width = img.shape[1]
        
        color = (0,255,0)
        
        if charge_battery < 14.0:
            color = (0,255,255)
        if charge_battery < 13.6:
            color = (0,0,255)
        
        battery_position = (width - 250, 100)
        battery_text = f"Copter: {charge_battery}V"
        img = self.print_text_to_img(img, battery_text, battery_position[0], battery_position[1], color)

        return img

    # Отрисовка режима полета
    def fly_mode_print(self, img, mode_id):
        height = img.shape[0]
        width = img.shape[1]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        mode = {
            0: "STABILIZE",
            1: "ACRO",
            2: "ALT_HOLD",
            3: "AUTO",
            4: "GUIDED",
            5: "LOITER",
            6: "RTL",
            7: "CIRCLE",
            9: "LAND",
            11: "DRIFT",
            13: "SPORT",
            16: "POSHOLD",
            17: "BRAKE",
            18: "THROW",
            19: "AVOID_ADSB",
            20: "GUIDED_NOGPS",
            # Добавьте другие режимы при необходимости
        }

        fly_mode_text = mode[mode_id]
        (text_width, _), _ = cv2.getTextSize(
            fly_mode_text, font, font_scale, font_thickness
        )
        fly_mode_position = (width // 2 - text_width // 2, height - 60)
        img = self.print_text_to_img(
            img, fly_mode_text, fly_mode_position[0], fly_mode_position[1]
        )

        return img

    # Отрисовка расстояния от точки старта
    def distance_print(self, img, distance_m):
        width, height = img.shape[1], img.shape[0]
        distance_position = (width - 120, height - 60)
        distance_text = f"D: {distance_m} M"
        img = self.print_text_to_img(img, distance_text, distance_position[0], distance_position[1])

        return img

    # Отрисовка высоты от точки старта
    def height_print(self, img, hight_m):
        height_img = img.shape[0]
        angles_position = (20, height_img - 60)
        height_text = f"H: {hight_m} M"
        img = self.print_text_to_img(img, height_text, angles_position[0], angles_position[1])

        return img
    
    # Нарисовать точку нормали
    def normal_point(self, img, pitch, roll):
        rf=1
        CaM = np.asarray([
        [225, 0, 325],              
        [0, 225, 230], 
        [0, 0, 1]
        ])
        
        tan_x = np.tan(roll)
        tan_y = np.tan(pitch)
        
        dpp = (int(tan_x*CaM[0,0] + CaM[0,2])*rf, int(tan_y*CaM[1,1] + CaM[1,2])*rf)
        
        cv2.circle(img, dpp, 16, (255, 255, 255), -1)
        cv2.circle(img, dpp, 10, (0, 102, 252), -1)
            
        return img

    # Отобразить инфрмацию на дисплей
    def show_info_on_display(self, img, msg_pix):

        if "ATTITUDE" in msg_pix:
            self.yaw = msg_pix["ATTITUDE"]["yaw"]
            self.pitch = msg_pix["ATTITUDE"]["pitch"]
            self.roll = msg_pix["ATTITUDE"]["roll"]
            img = self.normal_point(img, self.pitch, self.roll)

        if "AHRS2" in msg_pix and "HOME_POSITION" in msg_pix:
            self.height = int(msg_pix["AHRS2"]["altitude"] - msg_pix["HOME_POSITION"]["altitude"] / 1000)

        if "SYS_STATUS" in msg_pix:    
            self.battery = round(msg_pix["SYS_STATUS"]["voltage_battery"] / 1000.0, 2)

        if "HEARTBEAT" in msg_pix:  
            if msg_pix["HEARTBEAT"]["custom_mode"] == 0:
                self.count_mode += 1
            else:
                self.count_mode = 0
                self.fly_mode = msg_pix["HEARTBEAT"]["custom_mode"]
            
            if msg_pix["HEARTBEAT"]["custom_mode"] == 0 and self.count_mode > self.max_count:
                self.fly_mode = msg_pix["HEARTBEAT"]["custom_mode"]
                
        
        if "LOCAL_POSITION_NED" in msg_pix:  
            self.local_distance = int(math.sqrt(msg_pix["LOCAL_POSITION_NED"]["x"] ** 2 + (msg_pix["LOCAL_POSITION_NED"]["y"]) ** 2))
        
        img = self.compas_print(img, self.yaw)
        img = self.battery_print(img, self.battery)
        img = self.height_print(img, self.height)
        img = self.distance_print(img, self.local_distance)
        img = self.fly_mode_print(img, self.fly_mode)

        return img