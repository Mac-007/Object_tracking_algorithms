# Object_detection_1.py
import os
import cv2
from ultralytics import YOLO


class VideoPersonDetector:
    def __init__(self, input_video="Sample_Video.mp4", output_video="Sample_Video_Detected.mp4", model_path="yolov8n.pt"):
        # --- Paths ---
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_path = os.path.join(self.base_dir, input_video)
        self.output_path = os.path.join(self.base_dir, output_video)

        # --- Load YOLO model ---
        self.model = YOLO(model_path)

        # --- Video Capture ---
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {self.input_path}")

        # --- Video Properties ---
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # --- Output Writer ---
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

    def get_video_info(self):
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "output_path": self.output_path
        }

    def detect_frame(self, frame):
        """Return YOLO detections for a single frame."""
        results = self.model.predict(source=frame, verbose=False, stream=True)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label_name = self.model.names[cls]
                if label_name != "person":
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])
                
        return detections

    def cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def get_video_stream(self):
        return self.cap, self.out
