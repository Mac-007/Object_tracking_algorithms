import os
import cv2
from ultralytics import YOLO
import sys

# --- Get absolute paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))  # directory where this script is saved
input_path = os.path.join(base_dir, "Sample_Video.mp4")
output_path = os.path.join(base_dir, "Sample_Video_Detected.mp4")

# --- Load YOLO model (silent mode) ---
model = YOLO("yolov8n.pt")

# --- Read input video ---
cap = cv2.VideoCapture(input_path)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#fourcc = cv2.VideoWriter_fourcc(*"avc1")
#fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

# Create output writer
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))



# --- Print video info ---
print("Video Information:")
print(f"  - Resolution  : {width}x{height}")
print(f"  - FPS          : {fps}")
print(f"  - Total Frames : {total_frames}")
print("  - Detecting    : person\n")

# --- Process frames ---
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Run YOLO detection silently
    results = model.predict(source=frame, verbose=False, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label_name = model.names[cls]

            # Only keep 'person' class
            if label_name != "person":
                continue

            # Get bounding box + confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Person {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #cv2.rectangle(image, pt1, pt2, color, thickness)

            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            #cv2.putText(image, text, org, fontFace, fontScale, color, thickness, lineType)

    # Write frame to output
    out.write(frame)

    # --- Show progress (same line update) ---
    progress = (frame_count / total_frames) * 100
    sys.stdout.write(f"\rProcessing frame {frame_count}/{total_frames}  ({progress:.1f}% done)")
    sys.stdout.flush()

# --- Cleanup ---
cap.release()
out.release()
print("\n✅ Human detection completed successfully!")
print(f"💾 Output video saved at: {output_path}")
