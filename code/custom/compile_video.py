import cv2
import os
import glob

image_folder = "videos/tracks_vid6"         # folder containing your images
output_video = "videos/result_vid6.avi"     # name of final video
fps = 30                        # custom frame rate

# Collect all images
images = sorted(
    glob.glob(os.path.join(image_folder, "*.jpg")) +
    glob.glob(os.path.join(image_folder, "*.png"))
)

if len(images) == 0:
    raise RuntimeError("No images found in folder!")

# Read first valid frame
first_frame = None
for img_path in images:
    first_frame = cv2.imread(img_path)
    if first_frame is not None:
        break

if first_frame is None:
    raise RuntimeError("Could not read any image!")

height, width, _ = first_frame.shape

print(f"Video resolution: {width}x{height}, FPS: {fps}")

# Use a more reliable codec
fourcc = fourcc = cv2.VideoWriter_fourcc(*"MJPG") # H.264

video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames safely
for img_path in images:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: Could not read {img_path}, skipping.")
        continue

    # Ensure consistent size
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height))

    video.write(frame)

video.release()
print(f"Video saved as: {output_video}")

