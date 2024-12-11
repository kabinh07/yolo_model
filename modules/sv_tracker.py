import numpy as np
import supervision as sv
from modules.model import YOLOModel
import os

class SVTracker(YOLOModel):
    def __init__(self, config):
        super().__init__(config)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxCornerAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.source_path = self.config.track.source
        self.config = self.tracker_config_normalizer(config)

    def __callback(self, frame: np.ndarray, _: int) -> np.ndarray:
        results = self.model(frame, conf = 0.7, vid_stride = 50, verbose = False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = self.box_annotator.annotate(
            frame.copy(), detections=detections)
        return self.label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
    
    def track(self):
        self.model.fuse()
        self.config.track['project'] = self.project_name
        save_path = self.create_sequential_folder('sv_track')
        sv.process_video(
            source_path=self.source_path,
            target_path=os.path.join(save_path, 'vehicle_tracking_output.mp4'),
            callback=self.__callback
        )