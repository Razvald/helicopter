#!/bin/bash

# Путь к conda инициализации
source /home/orangepi/miniconda3/etc/profile.d/conda.sh

# Активация нужного окружения
conda activate baseto

cd /home/orangepi/SD/hp5

# Запуск программы
screen -dmS odom python3 /home/orangepi/SD/hp5/main.py
