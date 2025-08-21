from ultralytics import YOLO

def load_dataset(config_path="data.yaml"):
    """
    Loads the YOLO dataset using the provided config file.
    
    Args:
        config_path (str): Path to dataset YAML file.

    Returns:
        YOLO object (model not trained yet, just prepared for training/evaluation).
    """
    # Load a pretrained YOLO model (small version for speed)
    model = YOLO("yolov8n.pt")  
    
    print(f"Dataset loaded from {config_path}")
    return model


if __name__ == "__main__":
    # Example usage
    dataset_yaml = "data.yaml"  # path to your dataset config
    model = load_dataset(dataset_yaml)

    # Train model
    model.train(data=dataset_yaml, epochs=50, imgsz=640)

    # Validate model
    metrics = model.val(data=dataset_yaml)

    # Run on test set (inference)
    results = model.predict(source="data/test/images", save=True)
