from ultralytics import YOLO, solutions
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

class YOLOModel:
    def __init__(self, config):
        self.config = config
        self.model = YOLO(self.config.model.model_dir)
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
        self.config.track['project'] = self.project_name
        save_path = os.path.join(self.project_name, 'custom_track')
        count = 1
        while True:
            if os.path.exists(save_path):
                count += 1
                save_path = os.path.join(self.project_name, f'custom_track{count}')
            else: 
                break
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # result = self.model.track(**self.config.track)
        cap = cv2.VideoCapture(self.config.track.source)
        assert cap.isOpened()
        count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if success:
                img = Image.fromarray(im0)
                results = self.model.track(im0, persist=True, verbose = False)
                draw = ImageDraw.Draw(img)
                for idx, box in zip(results[0].boxes.id, results[0].boxes.xyxy):
                    draw.rectangle(box.numpy(), outline = 'red', width=2)
                    draw.text((box[0], box[1]), str(idx.item()), font=self.font)
                img.save(os.path.join(save_path, f'image_{count}.jpg'))
                count += 1
            else:
                break    
        cap.release()
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
    
    def __tracker_config_customizer(self):
        if self.config.track.