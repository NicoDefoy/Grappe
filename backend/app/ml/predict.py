import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from .model import GrapeClassifier

def load_trained_model(model_path="grape_classifier.pth"):
    """Charge le modèle entraîné."""
    model = GrapeClassifier(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image_input):
    """Fait une prédiction sur une image.
    Args:
        model: Le modèle entraîné
        image_input: Soit un chemin vers l'image, soit un fichier uploadé
    """
    # Charger l'image selon le type d'input
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = Image.open(image_input).convert('RGB')
    
    # Prétraiter l'image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Faire la prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Convertir les labels en noms de classes
    class_names = {
        0: "Non mature",
        1: "En maturation",
        2: "Mature"
    }
    
    return {
        'predicted_class': class_names[predicted.item()],
        'confidence': probabilities[0][predicted.item()].item(),
        'all_probabilities': {
            class_names[i]: prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }

def main():
    # Chemin vers le modèle et l'image à tester
    model_path = "grape_classifier.pth"
    image_path = input("Entrez le chemin vers l'image à classifier: ")
    
    # Charger le modèle
    print("Chargement du modèle...")
    model = load_trained_model(model_path)
    
    # Faire la prédiction
    print("Analyse de l'image...")
    result = predict_image(model, image_path)
    
    # Afficher les résultats
    print("\nRésultats de la prédiction:")
    print(f"Classe prédite: {result['predicted_class']}")
    print(f"Confiance: {result['confidence']*100:.2f}%")
    print("\nProbabilités pour chaque classe:")
    for class_name, prob in result['all_probabilities'].items():
        print(f"{class_name}: {prob*100:.2f}%")

if __name__ == "__main__":
    main() 