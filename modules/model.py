from ultralytics import YOLO, solutions
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import numpy as np
import random
from collections import Counter
import json

class YOLOModel:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(self.config.model.model_dir)
        self.classes = self.model.names
        self.data_dir = f'{self.config.model.data_dir}'
        self.project_name = os.path.join('runs', self.config.model.project_name)
        self.font_dir = '/home/kabin/Polygon/github/yolo_model/yolo_model/fonts/Arial.ttf'
        self.font = ImageFont.truetype(self.font_dir, 20)
    
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
        self.model.fuse()
        self.config.track['project'] = self.project_name
        if not self.config.track.track_id:
            self.tracker_config_normalizer()
            self.model.track(**self.config.track)
        else:
            class_colors = {class_name: tuple(random.randint(0, 150) for _ in range(3)) for class_name in self.classes}
            save_path = self.create_sequential_folder('track')
            cap = cv2.VideoCapture(self.config.track.source)
            self.tracker_config_normalizer()
            assert cap.isOpened()
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            video_writer = cv2.VideoWriter(os.path.join(save_path, 'vehicle_tracking_output.avi'), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            while cap.isOpened():
                success, im0 = cap.read()
                if success:
                    img = Image.fromarray(im0)
                    results = self.model.track(im0, persist=True, **self.config.track)
                    draw = ImageDraw.Draw(img)
                    for idx, box, cls in zip(results[0].boxes.id, results[0].boxes.xyxy, results[0].boxes.cls):
                        draw.rectangle(box.numpy(), outline = class_colors[int(cls.item())], width=2)
                        draw.text((box[0], box[1]), str(int(idx.item())), font=self.font)
                        draw.text((box[0], box[1]+20), str(self.classes[int(cls.item())]), font=self.font)
                    video_writer.write(np.array(img))
                else:
                    break    
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
        return
    
    def count(self):
        cap = cv2.VideoCapture(self.config.track.source)
        assert cap.isOpened()
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        region_points = [(0, 0), (w, 0), (w, h), (0, h)]
        counter = solutions.ObjectCounter(
            region = region_points, 
            model = self.config.model.model_dir, 
        )
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print('Complete')
                break
            im0 = counter.count(im0)
            print(counter.classwise_counts)
        cap.release()
        cv2.destroyAllWindows()
        return
    
    def analysis(self):
        train_label_path = os.path.join(self.config.model.data_dir, 'train/labels')
        val_label_path = os.path.join(self.config.model.data_dir, 'val/labels')
        train_labels = os.listdir(train_label_path)
        val_labels = os.listdir(val_label_path)
        train_labels_list = []
        val_labels_list = []
        label_map = self.model.names
        for label in train_labels:
            with open(os.path.join(train_label_path, label)) as f:
                file = f.read().split('\n')
                file = [f for f in file if not f == '']
            for row in file:
                cls = int(row.split(' ')[0])
                train_labels_list.append(label_map[cls])
        for label in val_labels:
            with open(os.path.join(val_label_path, label)) as f:
                file = f.read().split('\n')
                file = [f for f in file if not f == '']
            for row in file:
                cls = int(row.split(' ')[0])
                val_labels_list.append(label_map[cls])
        train_labels_count = Counter(train_labels_list)
        val_labels_count = Counter(val_labels_list)
        counts = {
            "train_labels_count": dict(train_labels_count),
            "val_labels_counts": dict(val_labels_count)
        }
        save_path = self.create_sequential_folder('analysis')
        with open(os.path.join(save_path, "counts.json"), 'w') as f:
            json.dump(counts, f)
        print(f"Labels class counts are saved in {save_path}")
        return

    def tracker_config_normalizer(self):
        config = {}
        ignore_list = ["source", "track_id", "sv_track", "save", "track_activation_threshold", "lost_track_buffer", "minimum_matching_threshold", "box_thickness", "label_size", "frame_rate"]
        for key, value in self.config.track.items():
            if key in ignore_list:
                continue
            config[key] = value
        self.config.track = config
        return
    
    def create_sequential_folder(self, filename):
        save_path = os.path.join(self.project_name, filename)
        count = 1
        while True:
            if os.path.exists(save_path):
                count += 1
                save_path = os.path.join(self.project_name, f'{filename}{count}')
            else: 
                break
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path
