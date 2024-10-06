if __name__ == '__main__':
    from ultralytics import YOLO
    import torch

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Path to the data.yaml file
    data_yaml = r'D:\@Code\Python\ProjectASL2\dataset\data.yaml'

    # Create and train the YOLO model with adjustments
    model = YOLO('yolov5m.pt').to(device)  # Load a pre-trained YOLOv5 small model and move it to GPU
    model.train(
        data=data_yaml, # yaml location
        epochs=30,      # how long you want to train model 1 epoch = 1 cycle of dataset
        imgsz=416,      # image size
        weight_decay=0.0005,  # prevent overfitting
        batch=16,       # controls how much data is processed at once.
        lr0=0.01,       # defines the starting rate of learning
        lrf=0.01,       # controls how the learning rate is adjusted over time.
    )
