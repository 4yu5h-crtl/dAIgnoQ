import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from PIL import Image
import google.generativeai as genai
from dAIgnoQ.app import config
from dAIgnoQ.app.utils.architectures import ResNetVQC
from dAIgnoQ.app.utils.xai_utils import generate_gradcam, get_gemini_explanation, setup_gemini

class MedicalImageClassifier:
    """
    Unified classifier class for Medical Images using PyTorch, QSVM, and hybrid models.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.model_type = None  # 'pytorch', 'qsvm', 'hybrid', 'qsvm_pipeline'
        self.gemini_api_key = None
        
        # QSVM components
        self.pca = None
        self.mins = None
        self.ranges = None
        self.states_train = None
        self.resnet = None
        self.n_qubits = 12
        
    def load_model(self, model_path, model_type='pytorch'):
        """
        Loads a pre-trained model.
        """
        self.model_type = model_type
        if model_type == 'pytorch':
            # Assuming it's a ResNet50 for now based on d3code
            from torchvision import models
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        elif model_type == 'qsvm':
            self.model = joblib.load(model_path)
        elif model_type == 'hybrid':
            # Placeholder for HybridModel/ResNetVQC loading logic
            self.model = ResNetVQC()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

    def load_qsvm_pipeline(self, pipeline_path: str):
        """Load complete QSVM pipeline"""
        try:
            checkpoint = torch.load(pipeline_path, map_location=self.device)
            self.model = checkpoint["svc"]
            self.pca = checkpoint["pca"]
            self.mins = checkpoint["mins"]
            self.ranges = checkpoint["ranges"]
            self.states_train = checkpoint["states_train"]
            self.n_qubits = checkpoint.get("n_qubits", 12)
            self.model_type = 'qsvm_pipeline'
            
            # Load ResNet backbone
            from torchvision import models
            self.resnet = models.resnet50(pretrained=False)
            self.resnet.fc = torch.nn.Identity()
            if config.RESNET50_PATH.exists():
                self.resnet.load_state_dict(torch.load(config.RESNET50_PATH, map_location=self.device), strict=False)
            self.resnet.to(self.device)
            self.resnet.eval()
            return True
        except Exception as e:
            raise e
            
    def predict(self, image: Image.Image):
        """
        Runs inference on an image.
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        if self.model_type == 'pytorch' or self.model_type == 'hybrid':
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
            return predicted_class.item(), confidence.item()
            
        elif self.model_type == 'qsvm_pipeline':
            # Complete QSVM prediction logic
            from torchvision import transforms
            from dAIgnoQ.app.utils.quantum_utils import scale_to_angles, compute_states, kernel_from_states, get_quantum_device
            
            transform = transforms.Compose([
                transforms.Resize(config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.resnet(input_tensor).cpu().numpy()
                
            # PCA projection
            X_pca = self.pca.transform(features)
            
            # Quantum processing
            X_angles = scale_to_angles(X_pca, self.mins, self.ranges)
            q_device = get_quantum_device(n_qubits=self.n_qubits)
            states_test = compute_states(X_angles, n_qubits=self.n_qubits, device=q_device)
            
            # Kernel and Prediction
            K_test = kernel_from_states(states_test, self.states_train)
            prediction = self.model.predict(K_test)[0]
            confidence = np.max(self.model.decision_function(K_test)) # simplistic
            
            return int(prediction), float(confidence)

        elif self.model_type == 'qsvm':
            img_array = np.array(image.resize(config.IMG_SIZE)).flatten().reshape(1, -1)
            prediction = self.model.predict(img_array)[0]
            try:
                confidence = np.max(self.model.predict_proba(img_array))
            except:
                confidence = 1.0
            return int(prediction), confidence
            
    def get_gradcam(self, image: Image.Image):
        """
        Generates Grad-CAM visualization.
        """
        if self.model_type in ['pytorch', 'hybrid']:
            # Assuming we target the last layer before pooling/fc for ResNet
            # This is a bit brittle, might need to be more specific based on model architecture
            if hasattr(self.model, 'layer4'):
                target_layer = self.model.layer4[-1]
                return generate_gradcam(self.model, image, target_layer, self.device)
        return None

    def setup_gemini(self, api_key: str):
        self.gemini_api_key = api_key
        setup_gemini(api_key)
        
    def get_explanation(self, image: Image.Image, prediction: str, confidence: float):
        if not self.gemini_api_key:
            return "Gemini API key not configured."
        return get_gemini_explanation(image, prediction, confidence)
