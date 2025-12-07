import cv2
import os
import numpy as np

video_path = "videos/5.mov"   # change this
output_dir = "/home/saksham/samsad/mtech-project/deep-learning/cricket-ball-detection/dataset/edge_fleet/25_nov_2025/vid5/train/images/"                  # folder to save frames

os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")


for frame_idx in range(total_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Failed to read frame {frame_idx}")
        continue
    filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(filename, frame)


cap.release()
print(f"Extracted {len(os.listdir(output_dir))} frames and saved to '{output_dir}'")
