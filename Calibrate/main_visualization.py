import json
import os
import info_on_display
import cv2


if __name__ == "__main__":
    # Путь к папке
    set_dir = '2024_12_15_15_31_8_num_3'
    
    # Получение всех файлов с расширением .json
    json_files = [f for f in os.listdir(set_dir) if f.endswith('.json')]
    
    # Сортировка файлов по имени
    json_files.sort()

    start = 1900
    count_json = 150
    
    info_pix = info_on_display.InfoPixhawkOnDisplay()
    
    # Iterate over files in the dataset directory
    for filename in json_files[start:start + count_json]:
        # Read the JSON file
        with open(f'{set_dir}/{filename}', 'r') as file:
            data = json.load(file)
            if 'GNRMC' in data:
                if data['GNRMC']['status'] == 'A':
                    img_path = set_dir + '/' + os.path.splitext(filename)[0] + '.jpg'
                    image = cv2.imread(img_path)
            img = info_pix.show_info_on_display(image, data)
            cv2.imshow("Copter", img)
            if cv2.waitKey(10) == ord('q'):
                break