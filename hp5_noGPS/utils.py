import os
import shutil

def get_drive_space(folder_path):
    total, used, free = shutil.disk_usage(folder_path)
    return total, used, free

def serialize(data):
    if isinstance(data, dict):
        return {key: serialize(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize(element) for element in data]
    elif isinstance(data, (bool, int, float, str)):
        return data
    else:
        return ''

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip if the file is a symbolic link
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size
