import os
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def analyze_dataset():
    # Chemins
    base_dir = Path(__file__).parent.parent.parent.parent
    image_dir = base_dir / "usable_images"
    annotation_dir = base_dir / "annotations"
    
    # Statistiques
    stats = {
        "total_images": 0,
        "good_grapes": 0,
        "bad_grapes": 0,
        "image_sizes": [],
        "annotations": {}
    }
    
    # Parcourir les images
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            stats["total_images"] += 1
            img_id = os.path.splitext(img_name)[0]
            
            # Vérifier la taille de l'image
            img_path = image_dir / img_name
            with Image.open(img_path) as img:
                stats["image_sizes"].append(img.size)
            
            # Par défaut, considérer comme mauvaise grappe
            stats["annotations"][img_id] = {"is_good": False}
            stats["bad_grapes"] += 1
    
    # Sauvegarder les annotations
    output_file = annotation_dir / "grape_annotations.json"
    with open(output_file, 'w') as f:
        json.dump(stats["annotations"], f, indent=2)
    
    # Afficher les statistiques
    print("\nStatistiques du dataset:")
    print(f"Nombre total d'images: {stats['total_images']}")
    print(f"Grappes bonnes: {stats['good_grapes']}")
    print(f"Grappes mauvaises: {stats['bad_grapes']}")
    
    # Afficher la distribution des tailles d'images
    if stats["image_sizes"]:
        widths, heights = zip(*stats["image_sizes"])
        plt.figure(figsize=(10, 5))
        plt.scatter(widths, heights, alpha=0.5)
        plt.title("Distribution des tailles d'images")
        plt.xlabel("Largeur (pixels)")
        plt.ylabel("Hauteur (pixels)")
        plt.savefig("image_sizes_distribution.png")
        plt.close()

if __name__ == "__main__":
    analyze_dataset() 