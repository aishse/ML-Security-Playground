import torch 
from torchvision.transforms import v2

def preprocess_image(image): 
    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    
    tensor = transform(image) 
    tensor = tensor.unsqueeze(0) 
    
    return tensor 


