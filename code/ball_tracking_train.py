from ultralytics import YOLO #type:ignore

# Load a pretrained YOLO11n model
model = YOLO("yolov11n.pt")
#model.ckpt_path = "runs/detect/train8/weights/best.pt"

train_results = model.train(
    data="/home/saksham/samsad/mtech-project/deep-learning/Cricket-Ball-Trajectory-Prediction/data.yaml",  # Path to dataset configuration file
    epochs=50,  # Number of training epochs
    device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    lr0=0.001
)