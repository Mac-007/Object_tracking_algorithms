# ByteTrack Person Tracking (Supervision Integration)

- This module implements **ByteTrack** for multi-person tracking in videos using **YOLOv8** for detection and the lightweight, pure-Python **Supervision** library for tracking.

- It serves as an alternative to the original [ByteTrack repository](https://github.com/FoundationVision/ByteTrack.git), which requires complex builds and dependencies (CMake, Visual Studio Build Tools).  

- Using **Supervision's ByteTrack** (`sv.ByteTrack()`), this implementation is **simple, cross-platform, and fast**, while maintaining robust multi-object tracking performance.

---

## 📂 Folder Structure
```
D:.
|   .gitignore
|   abcd.txt
|   LICENSE
|   Object_detection.py
|   Object_detection_1.py
|   Readme.md
|   Sample_Video.mp4
|   Sample_Video_Detected.mp4
|   Sample_Video_Tracked.mp4
|   Sample_Video_Tracked_ByteSORT.mp4
|   Sample_Video_Tracked_DeepSORT.mp4
|   Sample_Video_Tracked_SORT.mp4
|   yolov8n.pt
|   
+---.vscode
|       settings.json
|       
+---Results
|   +---Tracking-by-Detection_TbD
|   |   +---DeepSORT
|   |   |       DeepSORT_1.jpg
|   |   |       DeepSORT_2.jpg
|   |   |       DeepSORT_3.jpg
|   |   |       DeepSORT_4.jpg
|   |   |       
|   |   \---SORT
|   |           SORT_1.jpg
|   |           SORT_2.jpg
|   |           SORT_3.jpg
|   |           SORT_4.jpg
|   |           
|   \---YOLO_Detection
|           YOLO_Detection_1.jpg
|           YOLO_Detection_2.jpg
|           YOLO_Detection_3.jpg
|           YOLO_Detection_4.jpg
|           
+---tracking_algorithms
|   \---Tracking-by-Detection_TbD
|       |   Readme.md
|       |   
|       +---ByteTrack
|       |       ByteSORT.py
|       |       Readme.md
|       |       yolov8n.pt
|       |       
|       +---DeepSORT
|       |       DeepSORT.py
|       |       Readme.md
|       |       yolov8n.pt
|       |       
|       \---SORT
|           |   Alex_Bewley_SORT.py
|           |   Readme.md
|           |   SORT.py
|           |   yolov8n.pt
|           |   
|           \---__pycache__
|                   Alex_Bewley_SORT.cpython-313.pyc
|                   
\---__pycache__
        Object_detection_1.cpython-313.pyc
```

---

## 🧠 Overview

| Component | Description |
|------------|--------------|
| **Object_detection_1.py** | Provides YOLOv8-based person detection. |
| **ByteSORT.py** | Tracks detected persons using Supervision’s ByteTrack. |
| **Supervision Library** | Supplies a simple, pure-Python `ByteTrack()` class that eliminates the need for complex C++ builds. |
| **Output Video** | `Sample_Video_Tracked_ByteSORT.mp4` – annotated with bounding boxes and person IDs. |

---
## Usage

#### Running the Tracker

```bash
python tracking_algorithms/Tracking-by-Detection_TbD/ByteSORT/ByteSORT.py
```

This will:

1. Load the YOLOv8 model (yolov8n.pt).

2. Detect persons in the input video.

3. Track them across frames using Supervision’s ByteTrack.

4. Display live tracking output.

5. Save the final video as `Sample_Video_Tracked_ByteSORT.mp4`.

--- 

## Example Output

**1. Video Information**
  - Resolution  : 640x360
  - FPS          : 29
  - Total Frames : 3493
  - Tracking     : person

**2. Frame-wise Performance (last frame)**
Frame 3493/3493 | YOLO: 12.63 FPS | ByteTrack: 453.80 FPS | Overall: 11.07 FPS

**3. Performance Summary**
  - Avg YOLO FPS     : 13.69
  - Avg ByteTrack FPS: 414.26
  - Avg Total FPS    : 11.65
  - Total frames     : 3493
  - Total time       : 306.21 sec
  - Total unique persons detected: 636

---

## Implementation Highlights

__ByteTrack via Supervision__

```
import supervision as sv
tracker = sv.ByteTrack()
```

This replaces the heavier:

```from yolox.tracker.byte_tracker import BYTETracker```

dependency used in the original ByteTrack repo.

Input Format
Each detection (from YOLO) is converted to:
```
sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)
```

and passed to:
```
tracker.update_with_detections(detections)
```

Outputs
- Bounding boxes with ID labels
- Real-time FPS display
- Processed video written to disk

---

## Why Use Supervision’s ByteTrack?
Feature	Original ByteTrack (GitHub)	Supervision ByteTrack
Build Process	Requires CMake + Visual Studio Build Tools	Pure Python (no build needed)
OS Compatibility	Limited (Linux-focused)	Works on Windows, macOS, Linux
Dependencies	Heavy (YOLOX, PyTorch, ONNX)	Light (NumPy, OpenCV)
Ease of Integration	Complex (multiple submodules)	Plug-and-play
Recommended for	Research / large-scale setups	Quick prototyping, education, production demos

The original ByteTrack repository (FoundationVision/ByteTrack
) provides the full training and deployment pipeline but can be challenging to build on Windows.
Using Supervision’s ByteTrack gives equivalent tracking performance for inference without compilation headaches.

---

## Sample Output Screenshot

(Optional: include a still frame showing bounding boxes and IDs here)
Example label overlay:

Always show details
Person | ID:42

Summary
Metric	Value
Detector	YOLOv8n (Ultralytics)
Tracker	ByteTrack (Supervision)
Input	Sample_Video.mp4
Output	Sample_Video_Tracked_ByteSORT.mp4
Total Persons Tracked	636
Avg Overall FPS	11.65
Total Runtime	306.21 sec

---

## References

- Ultralytics YOLOv8 Documentation

- Supervision Library (Roboflow)

- ByteTrack Original Repository (FoundationVision)

---

## Author / Contact

- **Author**: `Dr. Amit Chougule, PhD` 

- ### Email: [amitchougule121@gmail.com](mailto:amitchougule121@gmail.com)
---