# Cricket Ball Detection using YOLOv11 + Domain Adaptation

**EdgeFleet.AI Assessment — Detection + Tracking Pipeline**

This project implements a complete, end-to-end pipeline for detecting a cricket ball in static-camera videos. Since the ball often appears extremely small (~1% of image area), and the EdgeFleet test videos differ significantly from public datasets, a **two-stage domain adaptation strategy** is used:

1. **Pretrain YOLOv11 on a public cricket-ball dataset (Kaggle).**
2. **Fine-tune on manually annotated frames extracted from selected EdgeFleet videos.**

The final model is evaluated on a separate set of unseen EdgeFleet videos using detection + tracking + trajectory visualization.

---

## Project Overview

### **Why This Approach?**

Detecting a cricket ball in long-range videos is challenging because:

* The ball is extremely small.
* Motion blur is significant during fast deliveries.
* Manually annotating all videos is infeasible.

To solve this, we use **transfer learning + domain adaptation**:

1. **Train on Kaggle data** → learn generic ball features.
2. **Annotate only a small subset** of EdgeFleet videos.
3. **Fine-tune** the pretrained model on the in-domain samples.
4. **Evaluate** on the remaining unseen videos.

---

## Dataset Structure

### **Public Dataset Used**

We use the Kaggle dataset:

```
kushagra3204/cricket-ball-dataset-for-yolo
```

Downloaded programmatically using `kagglehub`.

Directory structure:

```
cricket_ball_data/
    train/
    val/
    test/
```

### **EdgeFleet Videos**

From the 15 provided videos:

#### **Used for Annotation + Fine-Tuning**

```
2, 3, 4, 5, 7, 9, 14
```

#### **Held-out for Final Evaluation**

```
6, 8, 10, 11, 12, 13, 15
```

---

## Pipeline Summary

### **1. Frame Extraction**

Selected videos are broken into individual frames:

```
video_x/
    images/
        frame_000001.png
        frame_000002.png
        ...
```

This makes annotation easier and avoids CVAT free-tier limitations.

---

### **2. Manual Annotation with CVAT**

* Single class: `ball`
* Bounding boxes created per frame
* CVAT interpolation used to reduce manual effort
* Exported in YOLO format

After export:

* Labels cleaned
* Extra columns removed
* Normalized format enforced:

  ```
  class x_center y_center width height
  ```

---

### **3. Quality Control**

Performed via simple Python utilities:

* Visualize bounding boxes over images
* Reconstruct annotated video sequences
* Check temporal alignment and detect annotation noise

Incorrect annotations were corrected or removed to ensure clean training data.

---

### **4. Model Training**

#### **Stage 1 — Pretraining**

YOLOv11 trained on Kaggle dataset.

#### **Stage 2 — Fine-tuning**

Model further trained on:

* Kaggle dataset
* Annotated EdgeFleet frames

This aligns the feature representation to the specific camera + pitch conditions.

---

### **5. Inference + Tracking**

For testing on unseen videos:

* Central region of the frame is cropped (ball usually near pitch).
* Frame is upscaled to improve tiny-ball detection.
* YOLOv11’s `track()` API used for temporal consistency.
* Trajectory drawn using last *k* centroids.
* FPS + movement angle displayed for debugging.
---

## Final Evaluation

Model performance is judged qualitatively on videos:

```
6, 8, 10, 11, 12, 13, 15
```

These videos were **never used** during annotation or training.
