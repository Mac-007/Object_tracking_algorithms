# tracking_algorithms/Tracking-by-Detection_TbD/DeepSORT/DeepSORT.py

import sys
import os
import time
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Add parent folder to sys.path to import Object_detection_1.py ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Object_detection_1 import VideoPersonDetector


class DeepSortPersonTracker:
    def __init__(self, input_video="Sample_Video.mp4", output_video="Sample_Video_Tracked.mp4"):
        # Initialize YOLO detector
        self.detector = VideoPersonDetector(
            input_video=input_video,
            output_video=output_video,
            model_path="yolov8n.pt"
        )
        self.cap, self.out = self.detector.get_video_stream()
        self.info = self.detector.get_video_info()

        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30, n_init=2, nms_max_overlap=1.0, max_cosine_distance=0.3)

    def run(self):
        print("Video Information:")
        print(f"  - Resolution  : {self.info['width']}x{self.info['height']}")
        print(f"  - FPS          : {self.info['fps']}")
        print(f"  - Total Frames : {self.info['total_frames']}")
        print("  - Tracking     : person\n")
        
        unique_ids = set()
        
        frame_count = 0
        start_time_total = time.time()

        # Timing lists
        yolo_times = []
        deepsort_times = []
        total_times = []

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

            # Convert to DeepSORT format ((x, y, w, h), conf, class)
            formatted_detections = []
            for x1, y1, x2, y2, conf in detections_xyxy:
                x, y = float(x1), float(y1)
                w, h = float(x2) - float(x1), float(y2) - float(y1)
                if w <= 0 or h <= 0:
                    continue
                formatted_detections.append(((x, y, w, h), float(conf), "person"))

            # --- DEEPSORT TRACKING ---
            start_deepsort = time.time()
            tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
            end_deepsort = time.time()
            deepsort_times.append(end_deepsort - start_deepsort)

            # --- DRAW RESULTS ---
            for track in tracks:

                #print(f"ID {track.track_id}, confirmed={track.is_confirmed()}, age={track.age}, hits={track.hits}")

                if not track.is_confirmed():
                    continue
                
                unique_ids.add(track.track_id)
                
                l, t, r, b = track.to_ltrb()
                track_id = track.track_id
                
                cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 1)
                cv2.putText(frame, f" Person | ID:{track_id}", (int(l), max(20, int(t) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # --- SHOW / SAVE FRAME ---
            self.out.write(frame)
            cv2.imshow("DeepSORT Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTracking stopped by user")
                break

            # --- TIME STATS ---
            frame_end = time.time()
            total_times.append(frame_end - frame_start)

            # Compute rolling averages
            avg_yolo_fps = 1 / np.mean(yolo_times[-30:]) if len(yolo_times) >= 1 else 0
            avg_deepsort_fps = 1 / np.mean(deepsort_times[-30:]) if len(deepsort_times) >= 1 else 0
            avg_total_fps = 1 / np.mean(total_times[-30:]) if len(total_times) >= 1 else 0

            sys.stdout.write(
                f"\rFrame {frame_count}/{self.info['total_frames']} | "
                f"YOLO: {avg_yolo_fps:.2f} FPS | "
                f"DeepSORT: {avg_deepsort_fps:.2f} FPS | "
                f"Overall: {avg_total_fps:.2f} FPS"
            )
            sys.stdout.flush()

        # --- SUMMARY ---
        total_elapsed = time.time() - start_time_total
        print("\n\nPerformance Summary:")
        print(f"  - Avg YOLO FPS     : {1 / np.mean(yolo_times):.2f}")
        print(f"  - Avg DeepSORT FPS : {1 / np.mean(deepsort_times):.2f}")
        print(f"  - Avg Total FPS    : {1 / np.mean(total_times):.2f}")
        print(f"  - Total frames     : {frame_count}")
        print(f"  - Total time       : {total_elapsed:.2f} sec")
        print(f"  - Total unique persons detected: {len(unique_ids)}")
        print("\nTracking completed successfully!")
        print(f"Output video saved at: {self.info['output_path']}")

        self.detector.cleanup()


if __name__ == "__main__":
    tracker = DeepSortPersonTracker(
        input_video="Sample_Video.mp4",
        output_video="Sample_Video_Tracked.mp4"
    )
    tracker.run()
