# Application de Classification de Grappes de Raisin 🍇

## C'est quoi ce projet ?
Cette application permet de déterminer si une grappe de raisin est :
- Mature (prête à être récoltée)
- En maturation (bientôt prête)
- Non mature (pas encore prête)

Elle utilise l'intelligence artificielle pour analyser les photos de grappes que vous lui montrez !

## Comment ça marche ?
1. Vous prenez une photo de grappe ou vous en téléchargez une
2. L'application analyse la photo
3. Elle vous dit dans quel état est la grappe
4. Elle vous montre aussi son niveau de confiance pour chaque état possible

## Comment installer l'application ?

### Ce dont vous avez besoin :
- Python version 3.8 ou plus récente
- Un terminal (Terminal sur Mac, PowerShell sur Windows)

### Les étapes d'installation :

1. **Créer un environnement Python** (c'est comme une boîte isolée pour notre application) :
   ```bash
   # Créer l'environnement
   python -m venv venv
   
   # L'activer :
   # Sur Mac/Linux :
   source venv/bin/activate
   # Sur Windows :
   venv\Scripts\activate
   ```

2. **Installer les outils nécessaires** :
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Lancer l'application** :
   ```bash
   streamlit run backend/app/web/interface.py
   ```

4. L'application s'ouvre automatiquement dans votre navigateur !

## Les fichiers du projet expliqués simplement

```
.
├── backend/                    # Tout le code de l'application
│   ├── app/                   # Le cœur de l'application
│   │   ├── ml/               # La partie intelligence artificielle
│   │   │   ├── model.py      # Le cerveau qui analyse les images
│   │   │   ├── dataset.py    # Gestion des images d'entraînement
│   │   │   ├── train.py      # Pour entraîner l'IA
│   │   │   ├── predict.py    # Pour faire les prédictions
│   │   │   └── preprocessing.py  # Préparation des images
│   │   └── web/              # L'interface que vous voyez
│   │       └── interface.py   # La page web de l'application
│   ├── grape_classifier.pth   # L'IA entraînée (le cerveau)
│   └── requirements.txt       # Liste des outils nécessaires
├── usable_images/            # Les images pour entraîner l'IA
└── annotations/             # Les étiquettes des images
```

## Comment on a créé cette application ?

1. **Préparation des données**
   - On a collecté beaucoup de photos de grappes
   - On a classé chaque grappe (mature, en maturation, non mature)
   - On a mis les photos dans `usable_images`
   - On a mis les classifications dans `annotations`

2. **Création de l'IA**
   - On a utilisé un modèle appelé ResNet50 (déjà expert en reconnaissance d'images)
   - On l'a adapté pour reconnaître spécifiquement les grappes
   - On l'a entraîné avec nos photos classées

3. **Création de l'interface**
   - On a utilisé Streamlit pour faire une jolie interface web
   - On a ajouté des boutons pour télécharger/prendre des photos
   - On a fait en sorte que les résultats soient faciles à comprendre

## Comment utiliser l'application ?

1. **Analyser une grappe**
   - Cliquez sur "Télécharger une image" ou "Prendre une photo"
   - Sélectionnez ou prenez une photo
   - Cliquez sur "Analyser l'image"
   - Attendez le résultat !

2. **Comprendre les résultats**
   - Vous verrez la classification principale (mature, en maturation, non mature)
   - Vous verrez aussi les pourcentages de confiance pour chaque possibilité
   - Plus le pourcentage est élevé, plus l'IA est sûre de sa réponse

## Besoin d'aide ?

Si vous rencontrez des problèmes :
1. Vérifiez que Python est bien installé
2. Vérifiez que vous avez activé l'environnement virtuel
3. Essayez de réinstaller les dépendances
4. N'hésitez pas à poser des questions !

## Pour aller plus loin

Si vous voulez améliorer l'application :
1. Ajouter plus d'images d'entraînement
2. Tester différents modèles d'IA
3. Ajouter de nouvelles fonctionnalités à l'interface 

## Les commandes importantes à connaître 🖥️

### Pour démarrer le projet

1. **Ouvrir le terminal**
   ```bash
   # Sur Mac : ouvrez "Terminal"
   # Sur Windows : ouvrez "PowerShell"
   ```

2. **Aller dans le dossier du projet**
   ```bash
   # Remplacez le chemin par celui de votre projet
   cd chemin/vers/Grappe
   ```

3. **Activer l'environnement virtuel**
   ```bash
   # Sur Mac/Linux :
   source venv/bin/activate
   
   # Sur Windows :
   venv\Scripts\activate
   
   # Vous devriez voir (venv) au début de votre ligne de commande
   ```

### Pour entraîner le modèle

1. **Lancer l'entraînement**
   ```bash
   # Assurez-vous d'être dans le dossier du projet
   python -m backend.app.ml.train
   ```

2. **Analyser le dataset**
   ```bash
   # Pour voir les statistiques sur vos images
   python -m backend.app.ml.analyze_dataset
   ```

### Pour lancer l'application

1. **Démarrer l'interface**
   ```bash
   # Depuis le dossier du projet
   streamlit run backend/app/web/interface.py
   ```

### Commandes utiles en cas de problème

1. **Réinstaller les dépendances**
   ```bash
   # Si vous avez des erreurs d'import
   pip install -r backend/requirements.txt
   ```

2. **Vérifier la version de Python**
   ```bash
   python --version
   # Doit afficher 3.8 ou plus
   ```

3. **Vérifier l'installation des packages**
   ```bash
   pip list
   # Doit montrer torch, streamlit, etc.
   ```

### Pour quitter

1. **Arrêter l'application**
   ```bash
   # Dans le terminal où l'application tourne
   Ctrl + C
   ```

2. **Désactiver l'environnement virtuel**
   ```bash
   deactivate
   ```

## Utilisation de Git 📚

### Première fois

1. **Cloner le projet**
   ```bash
   # Télécharger le code
   git clone https://github.com/NicoDefoy/Grappe.git
   
   # Aller dans le dossier
   cd Grappe
   ```

2. **Configurer l'environnement**
   ```bash
   # Créer et activer l'environnement virtuel
   python -m venv venv
   source venv/bin/activate  # (Mac/Linux)
   # ou
   venv\Scripts\activate     # (Windows)
   
   # Installer les dépendances
   pip install -r backend/requirements.txt
   ```

### Pour les mises à jour

1. **Avant de commencer à travailler**
   ```bash
   # Récupérer les dernières modifications
   git pull origin main
   ```

2. **Pendant le travail**
   ```bash
   # Voir les fichiers modifiés
   git status
   
   # Voir les modifications en détail
   git diff
   ```

3. **Pour sauvegarder les modifications**
   ```bash
   # Ajouter les modifications
   git add .
   
   # Créer un commit avec un message explicatif
   git commit -m "Description des modifications"
   
   # Envoyer sur GitHub
   git push origin main
   ```

### Commandes Git utiles

- **Voir l'historique**
  ```bash
  git log --oneline  # Version courte
  git log           # Version détaillée
  ```

- **Annuler des modifications**
  ```bash
  # Annuler les modifications non commitées
  git restore <nom-fichier>
  
  # Annuler le dernier commit
  git reset --soft HEAD~1
  ```

- **Créer une nouvelle branche**
  ```bash
  # Créer et aller sur la nouvelle branche
  git checkout -b nom-de-la-branche
  
  # Revenir sur main
  git checkout main
  ```

### ⚠️ Important
- Ne jamais commit les fichiers sensibles (ils sont dans .gitignore) :
  - Le modèle entraîné (*.pth)
  - Les images d'entraînement
  - Les annotations
  - Les dossiers __pycache__
  - L'environnement virtuel (venv) 