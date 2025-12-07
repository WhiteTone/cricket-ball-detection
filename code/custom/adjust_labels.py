import os
import glob

ANN_DIR = "/home/saksham/samsad/mtech-project/deep-learning/Cricket-Ball-Trajectory-Prediction/dataset/edge_fleet/25_nov_2025/vid2/train/labels/"   # <-- change this

files = glob.glob(os.path.join(ANN_DIR, "frame_*.txt"))

for fpath in files:
    new_lines = []
    
    with open(fpath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                parts = parts[:5]     # keep only first 5 columns
            new_lines.append(" ".join(parts))

    with open(fpath, "w") as f:
        f.write("\n".join(new_lines))

print("Done. All last columns removed.")
