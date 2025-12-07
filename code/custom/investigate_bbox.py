import cv2
import os
import glob


IMAGE_DIR = "/home/saksham/samsad/mtech-project/deep-learning/cricket-ball-detection/dataset/edge_fleet/25_nov_2025/all_cropped/train/images"           # folder containing .png images
LABEL_DIR = "/home/saksham/samsad/mtech-project/deep-learning/cricket-ball-detection/dataset/edge_fleet/25_nov_2025/all_cropped/train/labels"           # folder containing .txt files (YOLO format)
OUT_DIR   = "/home/saksham/samsad/mtech-project/deep-learning/cricket-ball-detection/dataset/edge_fleet/25_nov_2025/all_cropped/train/visualized_gt"    # output folder

os.makedirs(OUT_DIR, exist_ok=True)
CLASS_NAMES = ["ball"]  


def draw_yolo_boxes(img, label_path):
    h, w = img.shape[:2]
    
    if not os.path.exists(label_path):
        return img  # no labels for this file

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        
        cls, xc, yc, bw, bh = parts
        cls = int(cls)
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

        # YOLO normalized → pixel coordinates
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw class name
        label = CLASS_NAMES[cls] if CLASS_NAMES and cls < len(CLASS_NAMES) else str(cls)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img


image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))

for img_path in image_paths:
    filename = os.path.basename(img_path).replace(".png", "")
    label_path = os.path.join(LABEL_DIR, filename + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load {img_path}")
        continue

    img_vis = draw_yolo_boxes(img, label_path)

    save_path = os.path.join(OUT_DIR, filename + ".png")
    cv2.imwrite(save_path, img_vis)
    print("Saved:", save_path)

print("Done! All visualized images saved to:", OUT_DIR)
