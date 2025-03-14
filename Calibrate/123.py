import pandas as pd

# Чтение данных из файла CSV
file_name = 'vio_results_comparison_1000.csv'
df = pd.read_csv(file_name)

# Приведение столбцов
df['Max Iters'] = df['Max Iters'].fillna('None')  # Преобразование NaN в 'None'

# Новый столбец для FPS
df['Avg FPS'] = None

# Перебор всех строк
for idx, row in df.iterrows():
    print(f"\nНабор параметров #{idx + 1}/{len(df)}")
    print(row[['Top_k', 'Detection Threshold', 'Max Iters', 'Rotation method', 'Trace depth', 'RMSE', 'Time']])
    
    # Сбор FPS от пользователя
    fps_values = input("Введите значения FPS через запятую (например: 5.3, 6.7, 7.1): ")
    fps_list = [float(fps.strip()) for fps in fps_values.split(',')]
    
    # Вычисление среднего FPS
    avg_fps = sum(fps_list) / len(fps_list)
    df.at[idx, 'Avg FPS'] = avg_fps
    print(f"Средний FPS для этого набора: {avg_fps:.2f}")

# Удаляем столбец Time, сохраняем файл
df = df.drop(columns=['Time'])
output_file = 'vio_results_updated.csv'
df.to_csv(output_file, index=False)
print(f"\nВсе параметры обновлены. Таблица сохранена в файл: {output_file}")
