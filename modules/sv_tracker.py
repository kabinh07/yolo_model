import numpy as np
import supervision as sv
from modules.model import YOLOModel
import os

class SVTrackerModel(YOLOModel):
    def __init__(self, config):
        super().__init__(config)
        self.tracker = sv.ByteTrack(
            track_activation_threshold = self.config.track.track_activation_threshold or 0.25,
            lost_track_buffer = self.config.track.lost_track_buffer or 30,
            minimum_matching_threshold = self.config.track.minimum_matching_threshold or 0.8,
            frame_rate = self.config.track.frame_rate or 30
            )
        self.box_annotator = sv.BoxAnnotator(thickness = self.config.track.box_thickness or 1)
        self.label_annotator = sv.LabelAnnotator()
        self.source_path = self.config.track.source
        self.skip_frames = self.config.track.vid_stride

    def __callback(self, frame: np.ndarray, _: int) -> np.ndarray:
        results = self.model(frame, **self.config.track)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {self.model.names[class_id]}"
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
        self.tracker_config_normalizer()
        sv.process_video(
            source_path=self.source_path,
            target_path=os.path.join(save_path, 'vehicle_tracking_output.mp4'),
            callback=self.__callback
        )