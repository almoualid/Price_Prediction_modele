import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("xgboost_modele_voiture2.pkl")

# Charger les encoders
le_pmain = joblib.load("le_pmain.pkl")
le_carburant = joblib.load("le_carburant.pkl")
le_etat = joblib.load("le_etat.pkl")
le_origine = joblib.load("le_origine.pkl")
le_transmission = joblib.load("le_transmission.pkl")
le_modele = joblib.load("le_modele.pkl")
le_marque = joblib.load("le_marque.pkl")

# (Optionnel) Charger le scaler
try:
    scaler = joblib.load("scascaler_voiture2.pkl")
    use_scaler = True
except:
    scaler = None
    use_scaler = False

# Interface Streamlit
st.title("🚗 Prédiction du prix d'une voiture")

# Champs d'entrée
marque = st.selectbox("Marque", le_marque.classes_)
modele = st.selectbox("Modèle", le_modele.classes_)
carburant = st.selectbox("Carburant", le_carburant.classes_)
etat = st.selectbox("État", le_etat.classes_)
origine = st.selectbox("Origine", le_origine.classes_)
transmission = st.selectbox("Transmission", le_transmission.classes_)
pmain = st.selectbox("Première main", le_pmain.classes_)

annee_modele = st.number_input("Année du modèle", min_value=1990, max_value=2025, step=1)
kilometrage = st.number_input("Kilométrage (km)", min_value=0.0)
nbr_porte = st.number_input("Nombre de portes", min_value=1.0)
puissance_fiscal = st.number_input("puissance Fiscal (CV)", min_value=1.0)

# Encodage des données texte
X = np.array([[
    le_marque.transform([marque])[0],
    le_modele.transform([modele])[0],
    le_carburant.transform([carburant])[0],
    le_etat.transform([etat])[0],
    le_origine.transform([origine])[0],
    le_transmission.transform([transmission])[0],
    le_pmain.transform([pmain])[0],
    annee_modele,
    kilometrage,
    nbr_porte,
    puissance_fiscal
]])

# Standardisation si scaler disponible
if use_scaler:
    X = scaler.transform(X)

# Prédiction
if st.button("Prédire le prix"):
    prix = model.predict(X)[0]
    st.success(f"💰 Prix estimé : {prix:,.2f} MAD")
