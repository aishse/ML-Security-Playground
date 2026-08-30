import gradio as gr 

from models.model_loader import load_model
from utils.predict import predict

models, weights = load_model()

def classify_image(image): 
    predictions = predict(image, models, weights) 
    for prediction in predictions: 
        return {
            prediction["label"]: prediction["probability"] for prediction in predictions
        }

demo = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(num_top_classes=5), 
    title="ML Security Playground",
    description="Explore adversarial ML attacks and defenses"
)

demo.launch()

