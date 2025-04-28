import streamlit as st
import joblib
import numpy as np
from bs4 import BeautifulSoup
import requests

# Configuration de la page
st.set_page_config(page_title="Prédiction Voiture", layout="wide")



# Charger le modèle
model = joblib.load("xgboost_modele_voiture1.pkl")

# Encoders
le_pmain = joblib.load("le_pmain.pkl")
le_carburant = joblib.load("le_carburant.pkl")
le_etat = joblib.load("le_etat.pkl")
le_origine = joblib.load("le_origine.pkl")
le_transmission = joblib.load("le_transmission.pkl")
le_modele = joblib.load("le_modele.pkl")
le_marque = joblib.load("le_marque.pkl")

# Scaler
try:
    scaler = joblib.load("scaler_voiture1.pkl")
    use_scaler = True
except:
    scaler = None
    use_scaler = False

# Fonction pour récupérer des annonces d'Avito
def get_avito_ads(modele):
    url = f"https://www.avito.ma/fr/casablanca/voitures_d_occasion-%C3%A0_vendre?model={modele}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    items = soup.find_all("a", class_="sc-1jge648-0 jZXrfL")  
    ads = []

    for item in items[:5]:  # Limiter à 5 annonces pour éviter surcharge
        title = item.find("p", class_="sc-1x0vz2r-0 iHApav")
        price = item.find("p", class_="sc-1x0vz2r-0 dJAfqm sc-b57yxx-3 eTHoJR")
        title_text = title.text.strip() if title else "Titre inconnu"
        price_text = price.text.strip() if price else "Prix inconnu"
        ads.append(f"📌 {title_text} - 💵 {price_text}")
    return ads

# Titre
st.title("🚗 Prédiction du prix d'une voiture au Maroc 2025")

# Formulaire en colonnes
col1, col2 = st.columns(2)

with col1:
    marque = st.selectbox("🛠️ Marque", le_marque.classes_)
    carburant = st.selectbox("⛽ Carburant", le_carburant.classes_)
    etat = st.selectbox("🔍 État", le_etat.classes_)
    annee_modele = st.number_input("📅 Année du modèle", min_value=1990, max_value=2025, step=1)
    kilometrage = st.number_input("🚗 Kilométrage (km)", min_value=0.0)
    puissance_fiscal = st.number_input("💥 Puissance Fiscal (CV)", min_value=1.0)  

with col2:
    modele = st.selectbox("🚘 Modèle", le_modele.classes_)
    transmission = st.selectbox("⚙️ Transmission", le_transmission.classes_)
    origine = st.selectbox("📦 Origine", le_origine.classes_)
    pmain = st.selectbox("📋 Première main", le_pmain.classes_)
    nbr_porte = st.number_input("🚪 Nombre de portes", min_value=1.0)
    
    

# Autres champs



# Prédiction
if st.button("🔮 Prédire le prix"):
    X = np.array([[le_marque.transform([marque])[0],
                   le_modele.transform([modele])[0],
                   le_carburant.transform([carburant])[0],
                   le_etat.transform([etat])[0],
                   le_origine.transform([origine])[0],
                   le_transmission.transform([transmission])[0],
                   le_pmain.transform([pmain])[0],
                   annee_modele,
                   kilometrage,
                   nbr_porte,
                   puissance_fiscal]])

    if use_scaler:
        X = scaler.transform(X)

    prix = model.predict(X)[0]
    st.success(f"💰 Prix estimé : {prix:,.2f} MAD")

    # Annonces similaires Avito
    st.markdown("### 🔍 Annonces similaires sur Avito")
    annonces = get_avito_ads(modele)
    for a in annonces:
        st.markdown(f"- {a}")
