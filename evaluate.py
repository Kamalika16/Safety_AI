from data_loader import load_dataset

def evaluate_model(config_path="safety_dataset.yaml"):
    """
    Evaluates YOLO model on validation and test sets.
    """
    model = load_dataset(config_path)

    # Validate on validation set
    metrics = model.val(data=config_path)
    print("Validation metrics:", metrics)

    # Run on test images
    results = model.predict(source="data/test/images", save=True)
    print("Predictions saved in runs/predict/")

if __name__ == "__main__":
    evaluate_model()
