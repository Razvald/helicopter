import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение данных из файла CSV
df = pd.read_csv('vio_results_updated.csv')

# Приведение столбцов
df['Max Iters'] = df['Max Iters'].fillna('None')  # Преобразование NaN в 'None'

plt.figure(figsize=(12, 6))

# График 1: Зависимость RMSE от Detection Threshold
plt.subplot(1, 3, 1)
sns.boxplot(data=df, x='Detection Threshold', y='RMSE', hue='Top_k', palette='viridis')
plt.title('Влияние Detection Threshold на RMSE при различных Top_k')
plt.xlabel('Detection Threshold')
plt.ylabel('RMSE')
plt.legend(title='Top_k')

# График 2: Влияние Trace Depth
plt.subplot(1, 3, 2)
sns.barplot(data=df, x='Trace depth', y='RMSE', hue='Rotation method', ci=None, palette='magma')
plt.title('Влияние Trace Depth на RMSE (по методам поворота)')
plt.xlabel('Trace Depth')
plt.ylabel('RMSE')
plt.legend(title='Rotation Method')

# График 3: Влияние Max Iters
plt.subplot(1, 3, 3)
sns.lineplot(data=df[df['Max Iters'] != 'None'], x='Max Iters', y='RMSE', hue='Detection Threshold', style='Top_k', markers=True, palette='coolwarm')
plt.title('Влияние Max Iters на RMSE (по Detection Threshold)')
plt.xlabel('Max Iters')
plt.ylabel('RMSE')
plt.legend(title='Detection Threshold / Top_k')

plt.tight_layout()
#plt.show()


# Поиск минимального значения RMSE
min_rmse = df['RMSE'].min()

# Устанавливаем порог RMSE (минимальный RMSE + N метр)
threshold_rmse = min_rmse + 300

# Фильтрация конфигураций с RMSE ≤ минимальное значение RMSE + N метр
df_acceptable = df[df['RMSE'] <= threshold_rmse]

# Если есть конфигурации с допустимой точностью
if not df_acceptable.empty:
    # Выбираем конфигурацию с максимальным FPS
    best_acceptable = df_acceptable.loc[df_acceptable['Avg FPS'].idxmax()]
    print(f"Лучшая конфигурация с минимальным RMSE (добавлен {threshold_rmse - min_rmse} метр): \n{best_acceptable}")
else:
    # Если нет конфигураций с RMSE ≤ минимальное значение, то выбираем конфигурацию с максимальным FPS
    best_overall = df.loc[df['Avg FPS'].idxmax()]
    print(f"Конфигурация с максимальным FPS (при RMSE > {threshold_rmse}): \n{best_overall}")

