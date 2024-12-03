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
        if not 'data' in self.config.train.keys():
            self.config.train['data'] = os.path.join(self.data_dir, 'data.yaml')
        self.config.train['project'] = self.project_name
        self.model.train(**self.config.train)
        return

    def predict(self):
        self.config.predict['project'] = self.project_name
        self.model.predict(**self.config.predict)
        return

    def track(self):
        self.config.track['project'] = self.project_name
        self.model.track(**self.config.track)
        return