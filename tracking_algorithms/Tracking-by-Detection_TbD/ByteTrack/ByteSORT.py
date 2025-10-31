"""
ByteSORT.py
-----------
Implementation of ByteTrack using Supervision library
for person tracking in videos with YOLOv8.

Author: Dr. Amit Chougule, PhD
Date: 2025-10-31

Description:
    This version uses the pure-Python ByteTrack tracker
    provided by the `supervision` library. It integrates
    YOLOv8 person detection (from Object_detection_1.py)
    with ByteTrack for multi-person tracking.

Dependencies:
    pip install ultralytics supervision opencv-python numpy
"""

import sys
import os
import time
import cv2
import numpy as np
import supervision as sv  # <— key library for ByteTrack

# --- Add parent folder to sys.path to import Object_detection_1.py ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Object_detection_1 import VideoPersonDetector


class SupervisionByteTrackPersonTracker:
    def __init__(self,
                 input_video="Sample_Video.mp4",
                 output_video="Sample_Video_Tracked_ByteSORT.mp4"):
        # Initialize YOLO detector
        self.detector = VideoPersonDetector(
            input_video=input_video,
            output_video=output_video,
            model_path="yolov8n.pt"
        )
        self.cap, self.out = self.detector.get_video_stream()
        self.info = self.detector.get_video_info()

        # Initialize Supervision ByteTrack tracker
        self.tracker = sv.ByteTrack()

    def run(self):
        print("Video Information:")
        print(f"  - Resolution  : {self.info['width']}x{self.info['height']}")
        print(f"  - FPS          : {self.info['fps']}")
        print(f"  - Total Frames : {self.info['total_frames']}")
        print("  - Tracking     : person\n")

        frame_count = 0
        unique_ids = set()

        start_time_total = time.time()
        yolo_times, bytesort_times, total_times = [], [], []

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            frame_start = time.time()

            # --- YOLO DETECTION ---
            start_yolo = time.time()
            detections_xyxy = self.detector.detect_frame(frame)
            end_yolo = time.time()
            yolo_times.append(end_yolo - start_yolo)

            if len(detections_xyxy) == 0:
                self.out.write(frame)
                continue

            # Convert detections to Supervision Detections format
            xyxy = np.array([d[:4] for d in detections_xyxy], dtype=np.float32)
            conf = np.array([d[4] for d in detections_xyxy], dtype=np.float32)
            class_id = np.zeros_like(conf, dtype=int)  # only "person"
            detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)

            # --- BYTETrack update ---
            start_bytesort = time.time()
            tracked_detections = self.tracker.update_with_detections(detections)
            end_bytesort = time.time()
            bytesort_times.append(end_bytesort - start_bytesort)

            # --- DRAW RESULTS ---
            for xyxy_box, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                if track_id is None:
                    continue
                x1, y1, x2, y2 = map(int, xyxy_box)
                unique_ids.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame, f"Person | ID:{track_id}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # --- SAVE FRAME ---
            self.out.write(frame)
            cv2.imshow("ByteSORT (Supervision) Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTracking stopped by user.")
                break

            # --- Timing stats ---
            frame_end = time.time()
            total_times.append(frame_end - frame_start)

            avg_yolo_fps = 1 / np.mean(yolo_times[-30:]) if yolo_times else 0
            avg_bytesort_fps = 1 / np.mean(bytesort_times[-30:]) if bytesort_times else 0
            avg_total_fps = 1 / np.mean(total_times[-30:]) if total_times else 0

            sys.stdout.write(
                f"\rFrame {frame_count}/{self.info['total_frames']} | "
                f"YOLO: {avg_yolo_fps:.2f} FPS | "
                f"ByteTrack: {avg_bytesort_fps:.2f} FPS | "
                f"Overall: {avg_total_fps:.2f} FPS"
            )
            sys.stdout.flush()

        # --- SUMMARY ---
        total_elapsed = time.time() - start_time_total
        print("\n\nPerformance Summary:")
        print(f"  - Avg YOLO FPS     : {1 / np.mean(yolo_times):.2f}")
        print(f"  - Avg ByteTrack FPS: {1 / np.mean(bytesort_times):.2f}")
        print(f"  - Avg Total FPS    : {1 / np.mean(total_times):.2f}")
        print(f"  - Total frames     : {frame_count}")
        print(f"  - Total time       : {total_elapsed:.2f} sec")
        print(f"  - Total unique persons detected: {len(unique_ids)}")
        print("\nTracking completed successfully!")
        print(f"Output video saved at: {self.info['output_path']}")

        self.detector.cleanup()


if __name__ == "__main__":
    tracker = SupervisionByteTrackPersonTracker(
        input_video="Sample_Video.mp4",
        output_video="Sample_Video_Tracked_ByteSORT.mp4"
    )
    tracker.run()
