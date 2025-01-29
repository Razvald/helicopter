import logging
import datetime
import os

class CopterLogger():
    
    def __init__(self, name_module, path_dir):
        self.logger = logging.getLogger(name_module)
        self.logger.setLevel(logging.INFO)
        
        path_dir_module = os.path.join(path_dir, name_module)
        # Создаем папку для логирования
        if not os.path.isdir(path_dir_module):
            os.makedirs(path_dir_module)
            
        # Настройка обработчика и форматировщика
        ct = datetime.datetime.now()
        timestr = f'{ct.year}_{ct.month}_{ct.day}_{ct.hour}_{ct.minute}_{ct.second}'
        count_files = len(os.listdir(path_dir_module)) + 1
        file_path = os.path.join(path_dir_module, f'module_{name_module}_{timestr}_num_{count_files}.log')
        handler = logging.FileHandler(file_path, mode='w')
        formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
        
        # добавление форматировщика к обработчику 
        handler.setFormatter(formatter)
        
        # добавление обработчика к логгеру
        self.logger.addHandler(handler)
        
    def info(self, message):
        self.logger.info(message)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def error(self, message):
        self.logger.warning(message)
        
    def critical(self, message):
        self.logger.critical(message)
        
    def exception(self, message):
        self.logger.exception(message, exc_info=True)
