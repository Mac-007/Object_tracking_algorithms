## SORT vs DeepSORT Analysis

### Performance Summary Comparison

| **Metric** | **Description** | **SORT** | **DeepSORT** |
|-------------|----------------|-----------|---------------|
| **Avg YOLO FPS** | Speed of detection (object inference rate) | 12.97 | 16.61 |
| **Avg Tracker FPS** | Tracker update rate (data association speed) | <span style="background-color: #ffeb3b; padding: 2px 6px; border-radius: 4px;">342.34</span> | 2.67 |
| **Avg Total FPS** | End-to-end video processing rate | 11.06 | 2.24 |
| **Total Frames** | Total processed frames | 3493 | 3493 |
| **Total Time (s)** | Overall runtime | 322.10 | 1564.60 |
| **Unique Persons Tracked** | Distinct tracked IDs | 767 | 388 |

### Speed
- **SORT** is much faster — roughly **5× faster** end-to-end than **DeepSORT** (11.06 vs 2.24 FPS).
- Its tracking update rate (**342 FPS**) is exceptionally fast due to the simplicity of its Kalman filter + Hungarian matching, making it faster.
- In contrast, **DeepSORT** performs additional **deep feature extraction** using a **CNN-based appearance model** for each detected object, which significantly increases computational load and reduces overall processing speed.

### Accuracy / Robustness
- **DeepSORT** is much slower but more robust, as it includes a **deep appearance descriptor** for re-identification, making it better at maintaining consistent IDs across frames.
- This explains the **lower number of unique IDs (388)** — fewer identity switches compared to **SORT’s 767**.

### Efficiency
- For **real-time or near real-time** use cases → **SORT** is preferred.
- For **offline, high-accuracy tracking** → **DeepSORT** is preferred.

### Key Takeaways
- **SORT:** Fast, lightweight, suitable for real-time applications (e.g., live surveillance).  
- **DeepSORT:** Slower but more accurate identity tracking, better for analysis tasks where ID persistence matters.
---