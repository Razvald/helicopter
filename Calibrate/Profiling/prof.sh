#!/bin/bash

# Указываем путь к корневой директории для профилирования
REPORT_DIR_HOME="$(realpath "./Profiling")"
REPORT_DIR="$REPORT_DIR_HOME/Reports"

# Проверяем, передан ли файл для профилирования
if [ -z "$1" ]; then
    echo "Ошибка: Укажите файл Python для профилирования."
    echo "Пример использования: ./Profiling/prof.sh script.py"
    exit 1
fi

# Путь к Python-файлу
SCRIPT_PATH="$(realpath "$1")"

# Проверяем, существует ли указанный файл
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Ошибка: Файл $SCRIPT_PATH не найден."
    exit 1
fi

# Создание папки для отчётов, если её нет
if [ ! -d "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR"
    echo "Создана папка для отчётов: $REPORT_DIR"
fi

# Запуск профилирования
echo "Профилирование файла: $SCRIPT_PATH"
nsys profile -o report_temp python3 "$SCRIPT_PATH"

# Поиск последнего созданного файла с расширением .nsys-rep
LATEST_REPORT=$(find . -type f -name "report_temp*.nsys-rep" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)

# Проверка, найден ли файл
if [ -n "$LATEST_REPORT" ]; then
    echo "Последний созданный отчет: $LATEST_REPORT"

    # Создание нового имени для отчёта на основе текущей даты и времени
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    NEW_REPORT_NAME="report_$TIMESTAMP.nsys-rep"
    NEW_REPORT_PATH="$REPORT_DIR/$NEW_REPORT_NAME"

    # Перемещение отчёта с новым именем
    mv "$LATEST_REPORT" "$NEW_REPORT_PATH"

    echo "Отчет перемещен в: $NEW_REPORT_PATH"

    # Открытие отчета
    nsys-ui "$NEW_REPORT_PATH"
else
    echo "Ошибка: Отчет не найден."
    exit 1
fi
