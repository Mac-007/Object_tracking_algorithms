## 🧠 SORT-based Person Tracking
File: SORT.py

- This script implements multi-person tracking using the classic SORT (Simple Online and Realtime Tracking) algorithm developed by `Alex Bewley`(https://github.com/abewley/sort) (Alex_Bewley_SORT.py). 

- It integrates YOLOv8 (for person detection) with SORT (for motion-based tracking) to perform efficient online tracking of multiple people in video sequences.

SORT (Simple Online and Realtime Tracking) is extremely lightweight:

- It only uses a Kalman Filter + IoU matching.
- No deep learning, no embeddings.
- It just updates track states from detections.


---
## 📁 File Overview

File                     | Description
--------------------------|----------------------------------------------------
SORT.py                   | Main script integrating YOLOv8 detection with the SORT tracker.It utilizes Alex_Bewley_SORT.py which is Original SORT algorithm. 
Alex_Bewley_SORT.py       | Original implementation of SORT by Alex Bewley (adapted for module import).
Object_detection_1.py      | YOLOv8-based person detection for each frame.
yolov8n.pt                 | Pre-trained YOLOv8 model used for object detection.

---

## 🧩 Algorithm Workflow

1. YOLOv8 Detection
   - Each video frame is passed through the YOLOv8 model.
   - Bounding boxes for the "person" class are extracted.

2. SORT Tracking
   - Detections are passed to the SORT tracker.
   - SORT maintains unique track IDs using a Kalman Filter and IoU-based data association.
   - Tracks are updated frame-by-frame in real time.

3. Visualization & Output
   - Each detected and tracked person is shown with a bounding box and unique ID.
   - A processed video file is saved showing tracked movements.

---

## ⚙️ Dependencies

Install the following packages before running:

``pip install ultralytics``
``pip install opencv-python``
``pip install numpy``
``pip install filterpy``

Note: Alex_Bewley_SORT.py uses filterpy for Kalman filtering.  
Ensure the file name is exactly "Alex_Bewley_SORT.py" (no spaces).

---
## ▶️ Usage

1. Folder Structure
Your project should look like this:

Project/
│
├── Object_detection_1.py
├── yolov8n.pt
│
└── tracking_algorithms/
    └── Tracking-by-Detection_TbD/
        └── SORT/
            ├── SORT.py
            └── Alex_Bewley_SORT.py

2. Running the Tracker

To run SORT tracking on a video:
```bash
python SORT.py
```
3. Parameters

Parameter       | Description                                                    | Default
----------------|----------------------------------------------------------------|----------
input_video     | Path to the input video file                                   | "Sample_Video.mp4"
output_video    | Output video file with tracked results                         | "Sample_Video_Tracked_SORT.mp4"
max_age         | Max frames to keep a track alive without new detections        | 5
min_hits        | Minimum detections before a new track is confirmed             | 2
iou_threshold   | Minimum IoU required to associate a detection with a track     | 0.3

---
## 🧾 Example Output

Video Information:
  - Resolution  : 640x360
  - FPS          : 29
  - Total Frames : 3493
  - Tracker      : SORT (Alex Bewley)

Performance Summary:
  - Avg YOLO FPS : 12.97
  - Avg SORT FPS : 342.34
  - Avg Total FPS: 11.06
  - Total frames : 3493
  - Total time   : 322.10 sec
  - Total unique persons tracked: 767

- Avg YOLO FPS: speed of detection
- Avg SORT FPS: speed of association (per frame, excluding detection)
- Avg Total FPS: true end-to-end video processing rate

✅ Video Output:
- Bounding boxes around persons
- Labels showing "Person | ID:<id>"
- Saved output file (e.g., Sample_Video_Tracked_SORT.mp4)

---
## 📊 Comparison with DeepSORT

Feature             | SORT                               | DeepSORT
--------------------|------------------------------------|----------------------------------
Tracking Basis      | Motion + IoU                       | Motion + Appearance (Re-ID)
Re-Identification   | ❌ No                              | ✅ Yes
Computational Load  | ⚡ Very Fast                       | 🧠 Heavier (uses CNN embeddings)
Recommended Use     | Real-time tracking (stable camera) | Multi-person tracking in dynamic scenes

---
## 🧠 Author & Credits

- Original SORT Algorithm: Alex Bewley (https://github.com/abewley/sort)  
- Detection Backbone: Ultralytics YOLOv8 (https://github.com/ultralytics/ultralytics)

---

**Author / Contact**

**Author**: `Dr. Amit Chougule, PhD` 

### 📧 Email: [amitchougule121@gmail.com](mailto:amitchougule121@gmail.com)
---