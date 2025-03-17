# Анализ и добавление FPS
file_name = 'vio_parameters_comparison.csv'
df = pd.read_csv(file_name)
df['Max Iters'] = df['Max Iters'].fillna('None')
df['Avg FPS'] = None

for idx, row in df.iterrows():
    print(f"Parameter set #{idx + 1}/{len(df)}")
    fps_values = input("Enter FPS values (e.g., 5.3, 6.7): ")
    fps_list = [float(fps) for fps in fps_values.split(',')]
    df.at[idx, 'Avg FPS'] = sum(fps_list) / len(fps_list)

df.drop(columns=['Time'], inplace=True)
df.to_csv('vio_results_updated.csv', index=False)
print("Results saved to 'vio_results_updated.csv'")