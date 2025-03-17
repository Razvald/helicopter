# 123.py

import pandas as pd

# Load data from CSV file
file_name = 'vio_results_comparison_1000.csv'
df = pd.read_csv(file_name)

# Fill NaN values in 'Max Iters' column with 'None'
df['Max Iters'] = df['Max Iters'].fillna('None')

# Add a new column 'Avg FPS'
df['Avg FPS'] = None

# Iterate over each row
for idx, row in df.iterrows():
    print(f"\nParameter set #{idx + 1}/{len(df)}")
    print(row[['Top_k', 'Detection Threshold', 'Max Iters', 'Rotation method', 'Trace depth', 'RMSE', 'Time']])
    
    # Collect FPS values from the user
    fps_values = input("Enter FPS values separated by commas (e.g., 5.3, 6.7, 7.1): ")
    fps_list = [float(fps.strip()) for fps in fps_values.split(',')]
    
    # Calculate average FPS
    avg_fps = sum(fps_list) / len(fps_list)
    df.at[idx, 'Avg FPS'] = avg_fps
    print(f"Average FPS for this set: {avg_fps:.2f}")

# Remove 'Time' column and save the file
df = df.drop(columns=['Time'])
output_file = 'vio_results_updated.csv'
df.to_csv(output_file, index=False)
print(f"\nAll parameters updated. Table saved to file: {output_file}")
