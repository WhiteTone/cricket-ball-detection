from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import os

SAVE_DIR = "videos/tracks_vid6"
os.makedirs(SAVE_DIR, exist_ok=True)

CROP_SCALE = 0.5          # take center 50% of frame
UPSCALE_FACTOR = 2        # upscale crop by 2× before detection
CONF_THRESHOLD = 0.1


def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        return math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
    return 90.0


class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)

    def pop(self):
        self.queue.popleft()
        
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)


def center_crop_and_upscale(frame, crop_scale=CROP_SCALE, upscale_factor=UPSCALE_FACTOR):
    h, w = frame.shape[:2]

    # compute crop dimensions
    new_w = int(w * crop_scale)
    new_h = int(h * crop_scale)

    cx1 = (w - new_w) // 2
    cy1 = (h - new_h) // 2
    cx2 = cx1 + new_w
    cy2 = cy1 + new_h

    cropped = frame[cy1:cy2, cx1:cx2]

    upscaled = cv2.resize(
        cropped, None,
        fx=upscale_factor, fy=upscale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    return upscaled, (cx1, cy1, cx2, cy2), upscale_factor


model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)

video_path = os.path.join('videos', '6.mov')
cap = cv2.VideoCapture(video_path)
ret = True

centroid_history = FixedSizeQueue(10)
start_time = time.time()
interval = 0.6
paused = False
angle = 0
prev_frame_time = 0
frame_no = 0


while ret:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- FPS CALC ----------------
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-8)
    prev_frame_time = new_frame_time
    fps = int(fps)

    # ---------------- QUEUE TIME POP ----------------
    current_time = time.time()
    if current_time - start_time >= interval and len(centroid_history) > 0:
        centroid_history.pop()
        start_time = current_time

    crop_img, (cx1, cy1, cx2, cy2), S = center_crop_and_upscale(frame)

    results = model.track(crop_img, persist=True, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    box = boxes.xyxy

    if len(box) != 0:
        for i in range(box.shape[0]):
            # Convert to normal floats (NO inplace ops on tensors)
            x1 = box[i][0].item()
            y1 = box[i][1].item()
            x2 = box[i][2].item()
            y2 = box[i][3].item()

            # scale back from upscaled crop
            x1 /= S
            y1 /= S
            x2 /= S
            y2 /= S

            # shift back to original frame coordinates
            x1 += cx1
            x2 += cx1
            y1 += cy1
            y2 += cy1

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2

            centroid_history.add((centroid_x, centroid_y))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 0, 255), -1)


    if len(centroid_history) > 1:
        points = list(centroid_history.get_queue())
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (255, 0, 0), 4)


    if len(centroid_history) > 1:
        pts = list(centroid_history.get_queue())
        x_diff = pts[-1][0] - pts[-2][0]
        y_diff = pts[-1][1] - pts[-2][1]

        if x_diff != 0:
            m1 = y_diff / x_diff
            if m1 == 1:
                angle = 90
            elif m1 != 0:
                angle = 90 - angle_between_lines(m1)

        # Predict next positions
        future_positions = [pts[-1]]
        for i in range(1, 5):
            future_positions.append(
                (pts[-1][0] + x_diff * i, pts[-1][1] + y_diff * i)
            )

        for i in range(1, len(future_positions)):
            cv2.line(frame, future_positions[i-1], future_positions[i], (0, 255, 0), 4)
            cv2.circle(frame, future_positions[i], 3, (0, 0, 255), -1)

    
    cv2.putText(frame, f"Angle: {angle:.2f}", (20, 25), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    frame_resized = cv2.resize(frame, (1000, 600))
    cv2.imshow("frame", frame_resized)

    save_path = os.path.join(SAVE_DIR, f"frame_{frame_no:05d}.jpg")
    cv2.imwrite(save_path, frame_resized)
    frame_no += 1

    key = cv2.waitKey(20)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord(' '):
        paused = not paused
        while paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                paused = False
            elif key == ord('q'):
                break

cap.release()
# cv2.destroyAllWindows()
