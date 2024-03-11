from yolov8tracker import TrackObject
input_video_path = "/data/video.mp4"
output_video_path =  "data/out.mp4"

obj = TrackObject(input_video_path,output_video_path)
obj.process_video()