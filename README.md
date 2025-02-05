### Сводка по проекту и задачам

**Описание проекта**  
Вы работаете над задачей, связанной с улучшением производительности для БПЛА, где основное внимание уделяется обработке данных GPS и визуальной одометрии (VIO). Проект использует технологию XFeat для сопоставления ключевых точек на изображениях, а также Python-библиотеки OpenCV и PIL для обработки изображений. В вашем распоряжении находится папка `Calibrate`, содержащая изображения, JSON-файлы с данными кадра, скрипты и модель нейронной сети. Ваш текущий фокус — изучение файлов `calibrate.ipynb` и `vio_ort.py`, а также реализация и отладка их работы.

**Структура данных и файлов**  
1. **Файлы и папки**:
   - JSON-файлы содержат телеметрические данные кадра (GPS, сенсоры, координаты).
   - Изображения `.jpg` соответствуют кадрам, для которых хранятся данные в JSON.
   - Параметры камеры с "рыбьим глазом" описаны в `fisheye_2024-09-18.json`.
   - `vio_ort.py` реализует класс VIO для обработки данных и вычисления положения.
   - `calibrate.ipynb` используется для визуализации траекторий GPS и VIO.

**Основные задачи**  
1. Изучение функционала `VIO.add_trace_pt` для обработки изображений и данных JSON, включая извлечение ключевых точек, сопоставление и расчёт координат.
2. Настройка `calibrate.ipynb` для корректной визуализации траекторий GPS и VIO.
3. Добавление обработчиков ошибок для отсутствующих или повреждённых данных.
4. Нормализация масштабов данных GPS и VIO для их корректного отображения на графиках.