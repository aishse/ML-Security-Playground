import torch 
from torchvision.models import resnet50, ResNet50_Weights


def load_model(): 
    weights = ResNet50_Weights.DEFAULT 
    model = resnet50(weights=weights)
    model.eval()
    return model, weights 

