import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image
import google.generativeai as genai
from typing import Tuple

try:
    from captum.attr import LayerGradCam, LayerAttribution
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def generate_gradcam(model, image: Image.Image, target_layer, device) -> np.ndarray:
    """
    Generate Grad-CAM visualization for a PyTorch model.
    Extracted from medical_image_classifier.py.
    """
    model.eval()
    
    # Preprocess image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    
    # Hook to get activations and gradients
    activations = {}
    gradients = {}
    
    def hook_fn(module, input, output):
        activations['value'] = output
    
    def hook_grad_fn(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]
    
    # Register hooks
    hook = target_layer.register_forward_hook(hook_fn)
    grad_hook = target_layer.register_full_backward_hook(hook_grad_fn)
    
    # Forward pass
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1)
    
    # Backward pass
    model.zero_grad()
    output[0, predicted_class].backward()
    
    # Get gradients and activations
    grads = gradients['value'][0]
    acts = activations['value'][0]
    
    # Global average pooling of gradients
    pooled_gradients = torch.mean(grads, dim=[1, 2])
    
    # Weight the feature maps
    for i in range(acts.size(0)):
        acts[i, :, :] *= pooled_gradients[i]
    
    # Average the weighted feature maps
    heatmap = torch.mean(acts, dim=0).detach().cpu().numpy()
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    
    # Convert to RGB colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap_rgb = cm.jet(heatmap)[:, :, :3]
    
    # Overlay heatmap on original image
    original_img = np.array(image.convert('RGB'))
    overlay = heatmap_rgb * 0.4 + original_img * 0.6
    overlay = np.uint8(overlay)
    
    # Remove hooks
    hook.remove()
    grad_hook.remove()
    
    return overlay

def generate_shap_explanation(model, image: Image.Image, device) -> np.ndarray:
    """
    Generate SHAP explanation for a PyTorch model.
    Extracted from medical_image_classifier.py.
    """
    if not SHAP_AVAILABLE:
        return np.array(image)
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate SHAP explanation
    explainer = shap.DeepExplainer(model, input_tensor)
    shap_values = explainer.shap_values(input_tensor)
    
    # Get values for class 1 (usually Unhealthy)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, :, :, 1]
    
    # Normalize for visualization
    sv_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
    sv_img = np.mean(sv_norm, axis=0)
    sv_img = cv2.applyColorMap(np.uint8(255*sv_img), cv2.COLORMAP_JET)
    sv_img = cv2.cvtColor(sv_img, cv2.COLOR_BGR2RGB)
    
    return sv_img

def setup_gemini(api_key: str):
    """
    Configure the Gemini API.
    """
    genai.configure(api_key=api_key)

def get_gemini_explanation(image: Image.Image, prediction: str, confidence: float) -> str:
    """
    Get explanation from Gemini Vision model.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Analyze this medical image and provide a detailed explanation based on the following:
        - Predicted class: {prediction}
        - Confidence: {confidence:.2%}
        
        Please provide:
        1. A brief explanation of what the image shows.
        2. Features supporting the classification of {prediction}.
        3. Potential considerations or limitations.
        
        Keep it clear and informative, but note that it is not a diagnosis.
        """
        
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error generating Gemini explanation: {str(e)}"
