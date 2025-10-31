"""
SORT.py
-------
Wrapper for the original SORT tracker (Alex Bewley implementation)
integrated with YOLOv8-based person detection (from Object_detection_1.py).

Author: Dr. Amit Chougule, PhD (adapted for Bewley SORT)
Date: 2025-10-31

Description:
    This script performs person tracking using:
        - YOLOv8 for object detection (via Object_detection_1.py)
        - Alex Bewley’s SORT tracker (Kalman + IOU-based)
    It outputs a processed video file with bounding boxes and assigned IDs.

Dependencies:
    pip install ultralytics opencv-python filterpy numpy scipy
"""

import sys
import os
import time
import numpy as np
import cv2

# --- Add parent path for Object_detection_1 import ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import YOLOv8 detector ---
from Object_detection_1 import VideoPersonDetector

# --- Import Alex Bewley’s SORT implementation ---
from Alex_Bewley_SORT import Sort


class SORTPersonTracker:
    def __init__(self, input_video="Sample_Video.mp4",
                 output_video="Sample_Video_Tracked_SORT.mp4",
                 max_age=30, min_hits=3, iou_threshold=0.3,
                 model_path="yolov8n.pt"):
        """Initialize YOLO detector and SORT tracker."""

        # Initialize YOLO detector
        self.detector = VideoPersonDetector(
            input_video=input_video,
            output_video=output_video,
            model_path=model_path
        )
        self.cap, self.out = self.detector.get_video_stream()
        self.info = self.detector.get_video_info()

        # Initialize SORT tracker from Alex Bewley’s implementation
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def run(self):
        """Main tracking loop."""
        print("Video Information:")
        print(f"  - Resolution  : {self.info['width']}x{self.info['height']}")
        print(f"  - FPS          : {self.info['fps']}")
        print(f"  - Total Frames : {self.info['total_frames']}")
        print("  - Tracker      : SORT (Alex Bewley)\n")

        frame_count = 0
        start_time_total = time.time()

        yolo_times, sort_times, total_times = [], [], []
        unique_ids = set()

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

            # Convert to SORT format [[x1,y1,x2,y2,score], ...]
            dets_for_sort = np.array(detections_xyxy, dtype=float) if len(detections_xyxy) > 0 else np.empty((0, 5))

            # --- SORT TRACKING ---
            start_sort = time.time()
            tracked_objects = self.tracker.update(dets_for_sort)
            end_sort = time.time()
            sort_times.append(end_sort - start_sort)

            # --- DRAW RESULTS ---
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = obj
                unique_ids.add(int(track_id))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                cv2.putText(frame, f" Person | ID:{int(track_id)}", (int(x1), max(20, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # --- SHOW / SAVE FRAME ---
            self.out.write(frame)
            cv2.imshow("SORT Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTracking stopped by user")
                break

            # --- PERFORMANCE STATS ---
            frame_end = time.time()
            total_times.append(frame_end - frame_start)

            avg_yolo_fps = 1 / np.mean(yolo_times[-30:]) if len(yolo_times) >= 1 else 0
            avg_sort_fps = 1 / np.mean(sort_times[-30:]) if len(sort_times) >= 1 else 0
            avg_total_fps = 1 / np.mean(total_times[-30:]) if len(total_times) >= 1 else 0

            sys.stdout.write(
                f"\rFrame {frame_count}/{self.info['total_frames']} | "
                f"YOLO: {avg_yolo_fps:.2f} FPS | "
                f"SORT: {avg_sort_fps:.2f} FPS | "
                f"Overall: {avg_total_fps:.2f} FPS"
            )
            sys.stdout.flush()

        # --- SUMMARY ---
        total_elapsed = time.time() - start_time_total
        print("\n\nPerformance Summary:")
        if len(yolo_times) > 0:
            print(f"  - Avg YOLO FPS : {1 / np.mean(yolo_times):.2f}")
        if len(sort_times) > 0:
            print(f"  - Avg SORT FPS : {1 / np.mean(sort_times):.2f}")
        if len(total_times) > 0:
            print(f"  - Avg Total FPS: {1 / np.mean(total_times):.2f}")
        print(f"  - Total frames : {frame_count}")
        print(f"  - Total time   : {total_elapsed:.2f} sec")
        print(f"  - Total unique persons tracked: {len(unique_ids)}")
        print("\nTracking completed successfully!")
        print(f"Output video saved at: {self.info['output_path']}")

        self.detector.cleanup()


if __name__ == "__main__":
    tracker = SORTPersonTracker(
        input_video="Sample_Video.mp4",
        output_video="Sample_Video_Tracked_SORT.mp4",
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        model_path="yolov8n.pt"
    )
    tracker.run()
