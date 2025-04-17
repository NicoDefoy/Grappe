import torch
import torch.nn as nn
from pathlib import Path
from .model import GrapeClassifier
from .dataset import create_dataloaders

def train_model():
    # Configuration
    base_dir = Path(__file__).parent.parent.parent.parent
    image_dir = base_dir / "usable_images"
    annotation_dir = base_dir / "annotations"
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    print(f"Base directory: {base_dir}")
    print(f"Image directory exists: {image_dir.exists()}")
    print(f"Annotation directory exists: {annotation_dir.exists()}")
    
    # Création des dataloaders
    train_loader, val_loader = create_dataloaders(
        image_dir=str(image_dir),
        annotation_dir=str(annotation_dir),
        batch_size=batch_size
    )
    
    print(f"Nombre d'images d'entraînement: {len(train_loader.dataset)}")
    print(f"Nombre d'images de validation: {len(val_loader.dataset)}")
    
    # Initialisation du modèle
    model = GrapeClassifier(num_classes=3)
    
    # Configuration de l'entraînement
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entraînement
    model.train_model(train_loader, criterion, optimizer, num_epochs)
    
    # Sauvegarde du modèle
    model.save_model("grape_classifier.pth")
    print("Modèle entraîné et sauvegardé avec succès!")

if __name__ == "__main__":
    train_model() 