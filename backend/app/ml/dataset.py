import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from pathlib import Path
from collections import Counter

class GrapeDataset(Dataset):
    """Dataset personnalisé pour les images de grappes de raisin."""
    
    def __init__(self, image_dir, annotation_file, transform=None, split='train'):
        """
        Args:
            image_dir (str): Dossier contenant les images
            annotation_file (str): Fichier JSON des annotations
            transform (callable, optional): Transformations à appliquer aux images
            split (str): 'train', 'valid' ou 'test'
        """
        self.image_dir = Path(image_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Charger les annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Créer la liste des images et leurs labels
        self.images = []
        self.labels = []
        
        # Parcourir les annotations et extraire les informations nécessaires
        for img_info in self.annotations['images']:
            img_path = self.image_dir / img_info['file_name']
            if img_path.exists():
                self.images.append(img_path)
                # Convertir les catégories en indices 0-based
                categories = [ann['category_id'] for ann in self.annotations['annotations'] 
                            if ann['image_id'] == img_info['id']]
                # Convertir les labels pour qu'ils commencent à 0
                label = (categories[0] - 1) if categories else 0
                self.labels.append(label)
        
        # Afficher la distribution des labels
        label_counts = Counter(self.labels)
        print(f"\nDistribution des labels pour {split}:")
        for label, count in sorted(label_counts.items()):
            print(f"Label {label}: {count} images")
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        """Retourne une paire (image, label)."""
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(image_dir, annotation_dir, batch_size=32, train_split=0.8):
    """
    Crée les dataloaders pour l'entraînement et la validation.
    
    Args:
        image_dir (str): Dossier contenant les images
        annotation_dir (str): Dossier contenant les fichiers d'annotations
        batch_size (int): Taille des batchs
        train_split (float): Proportion des données pour l'entraînement
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Transformations pour l'augmentation des données d'entraînement
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Transformations pour la validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Créer les datasets
    train_dataset = GrapeDataset(
        image_dir=image_dir,
        annotation_file=os.path.join(annotation_dir, 'mimc_train_images.json'),
        transform=train_transform,
        split='train'
    )
    
    val_dataset = GrapeDataset(
        image_dir=image_dir,
        annotation_file=os.path.join(annotation_dir, 'mimc_valid_images.json'),
        transform=val_transform,
        split='valid'
    )
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader 