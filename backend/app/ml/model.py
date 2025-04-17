import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class GrapeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(GrapeClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        # Configuration des transformations d'images
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        return self.model(x)
        
    def predict(self, image_path):
        """
        Prédit la classe d'une image de raisin.
        Args:
            image_path (str): Chemin vers l'image à classifier
        Returns:
            int: Classe prédite (0, 1 ou 2)
        """
        self.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self(image)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()
    
    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        """
        Entraîne le modèle sur un ensemble de données.
        Args:
            train_loader (DataLoader): Chargeur de données d'entraînement
            criterion: Fonction de perte
            optimizer: Optimiseur
            num_epochs (int): Nombre d'époques d'entraînement
        """
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
    
    def save_model(self, path):
        """
        Sauvegarde le modèle entraîné.
        Args:
            path (str): Chemin où sauvegarder le modèle
        """
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """
        Charge un modèle pré-entraîné.
        Args:
            path (str): Chemin vers le modèle à charger
        """
        self.load_state_dict(torch.load(path))

def create_model():
    """Crée et initialise le modèle."""
    model = GrapeClassifier(3)
    return model

def load_model(path):
    """Charge un modèle sauvegardé."""
    model = create_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict(model, image_tensor):
    """
    Fait une prédiction sur une image.
    
    Args:
        model: Le modèle GrapeNet
        image_tensor: Tensor PyTorch de l'image (1, 3, 224, 224)
    
    Returns:
        Tuple (classe prédite, probabilités)
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
    return predicted_class, probabilities[0].tolist() 