from PIL import Image 
from models.model_loader import load_model 
from utils.predict import predict

model, weights = load_model() 

image = Image.open("/Users/anishkachauhan/project-sprint1/ML-Security-Playground/examples/tabby.png")

predictions = predict(image, model, weights) 

for prediction in predictions: 
    print(f"Prediction: {prediction['label']}")
    print(f"Confidence: {prediction['probability']:.2%}")
    print()
