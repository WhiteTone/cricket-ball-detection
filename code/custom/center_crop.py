import cv2
import os


input_dir = "/home/saksham/samsad/mtech-project/deep-learning/cricket-ball-detection/dataset/edge_fleet/25_nov_2025/all/valid/images"
output_dir = "/home/saksham/samsad/mtech-project/deep-learning/cricket-ball-detection/dataset/edge_fleet/25_nov_2025/all_cropped/valid/images"
crop_ratio = 0.5   # 0.5 = crop center 50% of width/height

os.makedirs(output_dir, exist_ok=True)

# Loop over all PNG/JPG images
for fname in os.listdir(input_dir):
    if not (fname.endswith(".png") or fname.endswith(".jpg")):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)

    if img is None:
        print("Failed to read:", img_path)
        continue

    H, W = img.shape[:2]
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)

    # Coordinates for center crop
    y1 = (H - crop_h) // 2
    y2 = y1 + crop_h
    x1 = (W - crop_w) // 2
    x2 = x1 + crop_w

    cropped = img[y1:y2, x1:x2]

    # Upsample back to original resolution
    upsampled = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_CUBIC)

    # Save processed image
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, upsampled)

print("Done center-cropping + upsampling all images!")
