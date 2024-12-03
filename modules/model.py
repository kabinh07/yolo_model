from ultralytics import YOLO
import os
import subprocess

class YOLOModel:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(self.config.model.model_dir)
        self.data_dir = f'{self.config.model.data_dir}'
        self.project_name = os.path.join('runs', self.config.model.project_name)
    
    def train(self):
        config = {}
        for key, value in self.config.train.items():
            if not value is None:
                config[key] = value
        if not 'data' in config.keys():
            config['data'] = os.path.join(self.data_dir, 'data.yaml')
        config['project'] = self.project_name
        print(config)
        self.model.train(**config)
        return

    def predict(self):
        config = {}
        for key, value in self.config.predict.items():
            if not value is None:
                config[key] = value
        config['project'] = self.project_name
        self.model.predict(**config)
        return

    def track(self):
        
        