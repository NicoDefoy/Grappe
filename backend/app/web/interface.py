import streamlit as st
import torch
from PIL import Image
import io
import sys
from pathlib import Path

# Ajouter le chemin du backend au PYTHONPATH
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

from app.ml.model import GrapeClassifier
from app.ml.predict import predict_image

# Configuration de la page
st.set_page_config(
    page_title="Classificateur de Grappes de Raisin",
    page_icon="🍇",
    layout="centered"
)

# Style personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #9ed5ac;
        color: #1a1a1a;
        border-radius: 10px;
        padding: 15px 30px;
        font-size: 20px;
        font-weight: bold;
        width: 100%;
        margin: 20px 0;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        outline: none !important;
    }
    .stButton>button:hover {
        background-color: #8bc49a;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stButton>button:focus {
        box-shadow: none !important;
        outline: none !important;
        border: none !important;
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        border: 1px solid #e9ecef;
    }
    h1 {
        color: #2c3e50;
        font-weight: bold;
    }
    p {
        color: #2c3e50;
        font-size: 16px;
    }
    .stProgress > div > div {
        background-color: #4a7c59;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre avec emoji et description
st.title("🍇 Classificateur de Grappes de Raisin")
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p style='font-size: 18px; color: #2c3e50;'>
            Cette application permet de classifier l'état de maturité des grappes de raisin.
            <br>Téléchargez une image et découvrez si la grappe est mature, en maturation ou non mature.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chargement du modèle
@st.cache_resource
def load_model():
    model = GrapeClassifier(num_classes=3)
    model.load_state_dict(torch.load("grape_classifier.pth"))
    model.eval()
    return model

# Choix de la méthode d'entrée
option = st.radio(
    "Comment souhaitez-vous fournir l'image ?",
    ["Télécharger une image", "Prendre une photo"],
    horizontal=True
)

image = None

if option == "Prendre une photo":
    image = st.camera_input("Prenez une photo de la grappe de raisin")
else:
    image = st.file_uploader("Téléchargez une image de grappe de raisin", type=["jpg", "jpeg", "png"])

if image is not None:
    # Affichage de l'image
    st.image(image, caption="Image analysée", use_column_width=True)
    
    # Bouton de prédiction avec style personnalisé
    if st.button("Analyser l'image", key="analyze"):
        with st.spinner("Analyse en cours..."):
            # Chargement du modèle
            model = load_model()
            
            # Prédiction
            result = predict_image(model, image)
            
            # Affichage des résultats dans une mise en page améliorée
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; margin: 20px 0;'>
                    <h2 style='color: #2c3e50; margin-bottom: 20px; font-size: 24px;'>Résultats de l'analyse</h2>
                    <div style='background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                        <p style='font-size: 20px; color: #2c3e50; margin-bottom: 10px;'>
                            État de la grappe : <strong style='color: #4a7c59;'>{}</strong>
                        </p>
                        <p style='font-size: 18px; color: #2c3e50;'>
                            Niveau de confiance : <strong style='color: #4a7c59;'>{:.2f}%</strong>
                        </p>
                    </div>
                </div>
                """.format(
                    result['predicted_class'],
                    result['confidence'] * 100
                ), unsafe_allow_html=True)
            
            # Affichage des probabilités avec style amélioré
            st.markdown("<h3 style='color: #2c3e50; margin: 20px 0;'>Probabilités pour chaque état :</h3>", unsafe_allow_html=True)
            for class_name, prob in result['all_probabilities'].items():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"<p style='color: #2c3e50; font-size: 16px; text-align: right;'>{class_name}</p>", unsafe_allow_html=True)
                with col2:
                    st.progress(prob)
                    st.markdown(f"<p style='color: #4a7c59; font-size: 14px; text-align: right;'>{prob*100:.2f}%</p>", unsafe_allow_html=True) 