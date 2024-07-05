from ultralytics import YOLO

def load_yolo_model(model_path):
    model = YOLO(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
