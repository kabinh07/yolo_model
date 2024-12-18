from ultralytics import YOLO

model = YOLO('models/best.pt')

model.predict('data/vehicle_detection/cctv_footage_frames/0_1.jpg', save = True)

print(model.names)