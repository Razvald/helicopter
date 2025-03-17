# Optimization Tests

## Цель тестов
Основная задача тестирования заключалась в определении оптимальной конфигурации параметров визуальной инерциальной одометрии (VIO) для дрона. Целью было повышение FPS без значительных потерь в точности, измеряемой через RMSE (среднеквадратичная ошибка).

## Подход к тестированию

### 1. Анализ оригинального кода
Сначала был проведён анализ оригинального файла `vio_ort.py`, чтобы выявить основные узкие места. Была оптимизирована работа с параметрами, которые потенциально влияли на производительность:
- **Top_k** — количество ключевых точек для сопоставления.
- **Detection Threshold** — порог детекции для алгоритма.
- **Max Iters** — максимальное количество итераций.
- **Rotation method** — метод обработки поворота изображения.
- **Trace depth** — глубина трейсинга.

### 2. Тестирование на компьютере
Для первичной оценки эффективности параметров был реализован скрипт `optimize_vio_performance.ipynb`, который тестировал каждую комбинацию параметров на заранее загруженных данных:
- Данные (изображения и JSON-файлы) были загружены из директории.
- Для каждой комбинации параметров рассчитывались значения RMSE и время обработки.
- Итоговые данные сохранялись в файл `vio_results_updated.csv`.

Однако результаты тестов на компьютере не соответствовали реальным условиям работы на дроне. Причины:
- Компьютер имеет значительно большую вычислительную мощность.
- Обработка файлов с диска отличается от обработки кадров, поступающих с камеры дрона.

### 3. Тестирование на дроне
Для проверки параметров в реальных условиях каждая конфигурация была протестирована вручную на дроне:
- Производилась настройка таблицы с параметрами.
- Проводились полёты с записью данных о времени обработки (FPS) и точности.
- Средний FPS фиксировался вручную для каждой конфигурации.

## Анализ результатов

### Построение графиков
Для визуального анализа влияния параметров на RMSE и FPS был реализован следующий скрипт:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение данных
df = pd.read_csv('vio_results_updated.csv')
df['Max Iters'] = df['Max Iters'].fillna('None')

# Построение графиков
plt.figure(figsize=(12, 6))

# Влияние Detection Threshold на RMSE
plt.subplot(1, 3, 1)
sns.boxplot(data=df, x='Detection Threshold', y='RMSE', hue='Top_k', palette='viridis')
plt.title('Влияние Detection Threshold на RMSE при различных Top_k')
plt.xlabel('Detection Threshold')
plt.ylabel('RMSE')
plt.legend(title='Top_k')

# Влияние Trace Depth
plt.subplot(1, 3, 2)
sns.barplot(data=df, x='Trace depth', y='RMSE', hue='Rotation method', ci=None, palette='magma')
plt.title('Влияние Trace Depth на RMSE (по методам поворота)')
plt.xlabel('Trace Depth')
plt.ylabel('RMSE')
plt.legend(title='Rotation Method')

# Влияние Max Iters
plt.subplot(1, 3, 3)
sns.lineplot(data=df[df['Max Iters'] != 'None'], x='Max Iters', y='RMSE', hue='Detection Threshold', style='Top_k', markers=True, palette='coolwarm')
plt.title('Влияние Max Iters на RMSE (по Detection Threshold)')
plt.xlabel('Max Iters')
plt.ylabel('RMSE')
plt.legend(title='Detection Threshold / Top_k')

plt.tight_layout()
plt.show()
```

### Выбор оптимальной конфигурации
На основе результатов был определён следующий алгоритм для поиска лучшей конфигурации:
1. Найти минимальное значение RMSE.
2. Установить пороговое значение RMSE (`минимальное RMSE + 300 м`).
3. Из всех конфигураций с RMSE ниже порога выбрать ту, которая имеет наибольший FPS.
4. Если подходящих конфигураций нет, выбрать конфигурацию с максимальным FPS независимо от RMSE.

Пример реализации:
```python
min_rmse = df['RMSE'].min()
threshold_rmse = min_rmse + 300
df_acceptable = df[df['RMSE'] <= threshold_rmse]

if not df_acceptable.empty:
    best_acceptable = df_acceptable.loc[df_acceptable['Avg FPS'].idxmax()]
    print(f"Лучшая конфигурация: \n{best_acceptable}")
else:
    best_overall = df.loc[df['Avg FPS'].idxmax()]
    print(f"Конфигурация с максимальным FPS: \n{best_overall}")
```

## Итоги
- Реальное тестирование на дроне выявило, что наилучшая конфигурация — это баланс между RMSE и FPS.
- Оптимальная конфигурация была выбрана на основе данных, собранных в полевых условиях.
- В финальной версии системы используется оптимизированный файл `vio_ort.py`, который позволяет дрону работать с достаточной точностью и FPS для выполнения полётов в реальном времени.