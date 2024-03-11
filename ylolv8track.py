
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from typing import List
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import argparse
import torch

parser = argparse.ArgumentParser(
                    prog='yolov8',
                    description='This program helps to track objects',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', required=True)

args = parser.parse_args()


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


def match_detections_with_tracks(detections: Detections, tracks: List[STrack]) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


class TrackObject():

    def __init__(self, input_video_path, output_video_path) -> None:

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = YOLO("yolov8x.pt").to(self.device)
        self.model.fuse()

        self.CLASS_NAMES_DICT = self.model.model.names
        self.CLASS_ID = [2, 3, 5, 7]

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.video_info = VideoInfo.from_video_path(self.input_video_path)
        self.generator = get_video_frames_generator(self.input_video_path)
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)

    def process_video(self):
        with VideoSink(self.output_video_path, self.video_info) as sink:
            for frame in tqdm(self.generator, total=self.video_info.total_frames):
                results = self.model(frame)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                mask = np.array([class_id in self.CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                tracks = self.byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )
                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                labels = [
                    f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]
                frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                sink.write_frame(frame)


if __name__ == "__main__":
    obj = TrackObject(args.input, args.output)
    obj.process_video()

