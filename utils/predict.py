import torch

from utils.preprocess import preprocess_image 

def predict(image, model, weights): 
    tensor = preprocess_image(image)
    
    transform = weights.transforms()
    
    model_input = transform(tensor)
    
    with torch.no_grad():
        output = model(model_input) 
        
    probabilities = torch.nn.functional.softmax(output[0], dim=0) 
    values,indices = torch.topk(probabilities, 5) 
     
    categories = weights.meta["categories"] 
    
    predictions = []
    
    for value, index in zip(values, indices): 
        predictions.append({
            "label": categories[index], 
            "probability": float(value)
        })
    
    return predictions

        