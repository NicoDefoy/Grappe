import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Définition des transformations pour les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner à 224x224
    transforms.ToTensor(),  # Convertir en tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Moyennes ImageNet
        std=[0.229, 0.224, 0.225]    # Écarts-types ImageNet
    )
])

def preprocess_image(image_path):
    """
    Prétraite une image pour l'inférence.
    
    Args:
        image_path: Chemin vers l'image
        
    Returns:
        Tensor PyTorch normalisé (1, 3, 224, 224)
    """
    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    
    # Appliquer les transformations
    image_tensor = transform(image)
    
    # Ajouter une dimension batch
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def get_prediction_text(class_id, probabilities):
    """
    Convertit la prédiction en texte explicatif.
    
    Args:
        class_id: ID de la classe prédite
        probabilities: Liste des probabilités pour chaque classe
        
    Returns:
        Dict avec la classe prédite et les probabilités
    """
    classes = {
        0: "Grappe immature",
        1: "Grappe en maturation",
        2: "Grappe mature"
    }
    
    # Formater les probabilités en pourcentages
    probs_percent = {classes[i]: f"{prob * 100:.1f}%" 
                    for i, prob in enumerate(probabilities)}
    
    return {
        "prediction": classes[class_id],
        "probabilities": probs_percent
    } 