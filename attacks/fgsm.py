import torch 
import torch.nn.functional as F

def fgsm_attack(model, image, label, epsilon, weights):
    image = image.clone().detach()
    image.requires_grad = True
    
    transform = weights.transforms()
    model_input = transform(image)
    
    # forward pass 
    output = model(model_input) 
    
    # calculate the loss 
    loss = F.cross_entropy(output, label)
    
    # zero gradients
    model.zero_grad() 
    
    # backprop for the loss
    loss.backward() 
    
    # gradient
    gradient = image.grad.data
    
    
    # perform gradient descent 
    perturbation = epsilon * gradient.sign() 
    adversarial_image = image + perturbation
    
    # clamp pixels between 0 and 1 
    adversarial_image = torch.clamp(adversarial_image, 0, 1)
    
    return adversarial_image.detach()

 
     