from ultralytics import YOLO
import os

def train_model():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # you can change to yolov8s.pt, yolov8m.pt etc.

    # Absolute path to dataset YAML
    config_path = os.path.join("data.yaml")

    # Training parameters
    epochs = 50
    imgsz = 640

    # Start training
    model.train(
        data=config_path, 
        epochs=epochs, 
        imgsz=imgsz,
        project="runs/train",   # results will be saved here
        name="safety_model"     # name of the experiment
    )

if __name__ == "__main__":
    train_model()
