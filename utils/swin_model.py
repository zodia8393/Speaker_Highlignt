import torch
from timm import create_model

def load_swin_model():
    model = create_model('swin_tiny_patch4_window7_224', pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model
