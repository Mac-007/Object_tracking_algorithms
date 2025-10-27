import os
import cv2
import sys
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

    def print_video_info(self):
        print("Video Information:")
        print(f"  - Resolution  : {self.width}x{self.height}")
        print(f"  - FPS          : {self.fps}")
        print(f"  - Total Frames : {self.total_frames}")
        print("  - Detecting    : person\n")

    def detect_persons(self):
        """Run YOLOv8 on the video and save annotated output."""
        self.print_video_info()

        frame_count = 0
        print("Starting detection...\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_count += 1

            # Run YOLO detection silently
            results = self.model.predict(source=frame, verbose=False, stream=True)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label_name = self.model.names[cls]

                    # Only detect "person" class
                    if label_name != "person":
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"Person {conf:.2f}"

                    # Draw detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Write to output
            self.out.write(frame)

            # Show live window
            cv2.imshow("Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nDetection stopped by user")
                break

            # Show progress
            progress = (frame_count / self.total_frames) * 100
            sys.stdout.write(f"\rProcessing frame {frame_count}/{self.total_frames}  ({progress:.1f}% done)")
            sys.stdout.flush()

        self.cleanup()

    def cleanup(self):
        """Release all resources."""
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("\nHuman detection completed successfully!")
        print(f"Output video saved at: {self.output_path}")


if __name__ == "__main__":
    detector = VideoPersonDetector(
        input_video="Sample_Video.mp4",
        output_video="Sample_Video_Detected.mp4",
        model_path="yolov8n.pt"
    )
    detector.detect_persons()
