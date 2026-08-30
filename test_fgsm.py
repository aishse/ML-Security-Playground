import torch 
from PIL import Image 

from models.model_loader import load_model
from utils.preprocess import preprocess_image
from utils.predict import predict 
from attacks.fgsm import fgsm_attack

# TODO make the model testing use a tensor too 

# loading model
model, weights = load_model()

# load image
image = Image.open("/Users/anishkachauhan/project-sprint1/ML-Security-Playground/examples/tabby.png").convert("RGB")

predictions = predict(image, model, weights)

original_prediction = predictions[0]

print("ORIGINAL")
for prediction in predictions: 
    print(f"Prediction: {prediction['label']}")
    print(f"Confidence: {prediction['probability']:.2%}")
    print()

image_tensor = preprocess_image(image)

# get predicted class index 
categories = weights.meta["categories"]

label_index = categories.index(original_prediction["label"])
label = torch.tensor([label_index])

# run FGSM 
epsilon = 0.01 

# get an adversarial tensor using the image, label, and epsilon
adversarial_tensor = fgsm_attack(
    model,
    image_tensor,
    label,
    epsilon, 
    weights
    
)

# convert adversarial tensor back to image 
adversarial_image = adversarial_tensor.squeeze(0)
adversarial_image = adversarial_image.permute(1,2,0)
adversarial_image = (adversarial_image.cpu().numpy())


adversarial_pil = Image.fromarray(
    (adversarial_image * 255).astype("uint8")
)


# put the adversarial image into the predict function to get the prediction
adversarial_predictions = predict(
    adversarial_pil, 
    model,
    weights
)

adversarial_prediction = adversarial_predictions[0]

print("ADVERSARIAL")
for prediction in adversarial_predictions: 
    print(f"Prediction: {prediction['label']}")
    print(f"Confidence: {prediction['probability']:.2%}")
    print()

# determine if attack succeded 
attack_succeded = (
    original_prediction["label"] != adversarial_prediction["label"]
)

print(f"\nAttack succeded: {attack_succeded}")

