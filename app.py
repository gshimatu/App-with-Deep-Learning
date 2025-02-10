import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

#  Définition des chemins corrects
MODEL_PATH = "../app/model.keras"      # Lien du model
SCALER_PATH = "../app/scalar.pkl"      # Lien du scaler
FEATURES_PATH = "../app/features.pkl"  # Lien de la Liste des colonnes attendues

# 🛠️ Vérification et chargement des fichiers nécessaires
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Le modèle '{MODEL_PATH}' n'existe pas ! Vérifiez le chemin.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error(f"❌ Le scaler '{SCALER_PATH}' n'existe pas ! Vérifiez le chemin.")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error(f"❌ Le fichier '{FEATURES_PATH}' contenant les colonnes attendues n'existe pas !")
    st.stop()

# Chargement du modèle, scaler et colonnes attendues
model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scalar = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    all_features = pickle.load(f)

# Sidebar avec navigation entre les pages
st.sidebar.title("📌 Navigation")
page = st.sidebar.selectbox("Choisissez une page", ["🏠 Accueil", "📖 Instructions"])

# Sidebar pour le nom de l'utilisateur et les infos
st.sidebar.title("💼 À propos de l'application")
user_name = st.sidebar.text_input("Entrez votre nom :", "")

st.sidebar.markdown("""
**Prédiction MBA**  
Cette application utilise un modèle d'intelligence artificielle pour prédire si un candidat est susceptible de poursuivre un MBA en fonction de plusieurs critères.
""")

# Affichage des membres du groupe avec photos
st.sidebar.markdown("### 👥 Membres de l'équipe")

team_members = [
    {"nom": "Mukenndi Joël", "fonction": "Expert en sécurité", "photo": "Joel.jpg"},
    {"nom": "Nkura Winne", "fonction": "Data analyste", "photo": "Winner.jpg"},
    {"nom": "Shimatu Gauthier", "fonction": "Developpeur", "photo": "Gauthier.jpg"},
    {"nom": "Tamundel Bénédicte", "fonction": "Consultante en sécurité", "photo": "Benedicte.jpg"},
    {"nom": "Tovo Henrietto", "fonction": "IT réseau", "photo": "Henrietto.jpg"}
]

for member in team_members:
    st.sidebar.image(f"../app/images/{member['photo']}", width=400)
    st.sidebar.write(f"**{member['nom']}** – {member['fonction']}")

# 📌 Ajout du Copyright en bas du Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 MBA Prediction App. Tous droits réservés.")

if page == "🏠 Accueil":
    # 🎓 Interface principale
    st.title("🎓 Prédiction : Voulez-vous poursuivre un MBA ?")
    st.write("Remplissez les informations ci-dessous pour obtenir une prédiction.")

    # Variables continues
    age = st.slider("📅 Âge", 18, 60, 25)
    undergrad_gpa = st.slider("🎓 GPA Undergraduate", 0.0, 4.0, 3.0, 0.1)
    work_experience = st.slider("💼 Années d'expérience", 0, 40, 5)
    salary_before_mba = st.number_input("💰 Salaire actuel ($)", min_value=10000, max_value=500000, value=50000)
    gmat_score = st.slider("📊 GRE/GMAT Score", 200, 800, 600)
    university_ranking = st.slider("🏛️ Classement Université", 1, 1000, 500)
    entrepreneurial_interest = st.slider("🚀 Intérêt pour l'entrepreneuriat (1-10)", 1, 10, 5)
    networking_importance = st.slider("🔗 Importance du Networking (1-10)", 1, 10, 5)
    expected_salary_post_mba = st.number_input("📈 Salaire attendu après MBA ($)", min_value=20000, max_value=1000000, value=100000)

    # Variables catégoriques
    gender = st.selectbox("👤 Genre", ["Male", "Female", "Other"])
    undergrad_major = st.selectbox("🎭 Filière Undergraduate", ["Business", "Engineering", "Arts", "Sciences", "Other"])
    job_title = st.selectbox("🛠️ Poste Actuel", ["Analyst", "Manager", "Director", "Entrepreneur", "Consultant", "Other"])
    management_experience = st.selectbox("📋 Expérience en Management", ["Yes", "No"])
    mba_funding = st.selectbox("💳 Source de Financement MBA", ["Self", "Company Sponsorship", "Loan", "Scholarship"])
    post_mba_role = st.selectbox("🏆 Post-MBA Role", ["Consulting", "Finance", "Tech", "Entrepreneurship", "Other"])
    location_pref = st.selectbox("🌍 Préférence de Localisation (Post-MBA)", ["North America", "Europe", "Asia", "Other"])
    reason_mba = st.selectbox("🎯 Motivation pour le MBA", ["Career Growth", "Entrepreneurship", "Skill Development", "Other"])
    mode_mba = st.selectbox("🏫 Format MBA", ["Online", "On-Campus"])

    # Bouton pour lancer la prédiction
    if st.button("🔮 Prédire"):
        user_input = {
            'Age': age,
            'Undergraduate GPA': undergrad_gpa,
            'Years of Work Experience': work_experience,
            'Annual Salary (Before MBA)': salary_before_mba,
            'GRE/GMAT Score': gmat_score,
            'Undergrad University Ranking': university_ranking,
            'Entrepreneurial Interest': entrepreneurial_interest,
            'Networking Importance': networking_importance,
            'Expected Post-MBA Salary': expected_salary_post_mba,
            'Gender': gender,
            'Undergraduate Major': undergrad_major,
            'Current Job Title': job_title,
            'Has Management Experience': management_experience,
            'MBA Funding Source': mba_funding,
            'Desired Post-MBA Role': post_mba_role,
            'Location Preference (Post-MBA)': location_pref,
            'Reason for MBA': reason_mba,
            'Online vs. On-Campus MBA': mode_mba
        }

        user_data_df = pd.DataFrame([user_input])
        user_data_df = pd.get_dummies(user_data_df)

        # Ajouter les colonnes manquantes
        for col in all_features:
            if col not in user_data_df:
                user_data_df[col] = 0  

        # Réordonner les colonnes
        user_data_df = user_data_df[all_features]

        # Standardisation
        user_data_scaled = scalar.transform(user_data_df)

        # Prédiction avec le modèle
        prediction = model.predict(user_data_scaled)

        # Affichage du message avec le nom de l'utilisateur
        title = "Monsieur" if gender == "Male" else "Madame"
        if user_name.strip():
            st.markdown(f"### ✨ {title} {user_name}, voici votre résultat :")

        # Affichage du résultat
        if prediction[0][0] > 0.5:
            st.success("✅ Vous êtes susceptible de poursuivre un MBA !")
        else:
            st.error("❌ Vous n'êtes pas susceptible de poursuivre un MBA.")

elif page == "📖 Instructions":
    # 📖 Page des instructions
    st.title("📖 Instructions d'utilisation")
    st.markdown("""
    ### 📝 Informations demandées :
    - **📅 Âge** : L'âge au moment de la décision de poursuivre un MBA.
    - **👤 Genre** : Homme, Femme, Autre.
    - **🎓 Majeure de premier cycle** : Domaine d'études du baccalauréat (Ingénierie, Commerce, Arts, Sciences...).
    - **📊 GPA de premier cycle** : Moyenne générale du premier cycle (0 à 4).
    - **💼 Années d'expérience professionnelle** : Nombre d'années d'expérience avant le MBA.
    - **🛠️ Titre du poste actuel** : Poste actuel (Analyste, Manager, Consultant...).
    - **💰 Salaire annuel avant MBA** : Salaire en USD avant le MBA.
    - **📋 Expérience en management** : Oui/Non.
    - **📈 Score GRE/GMAT** : Score au test standardisé GRE ou GMAT.
    - **🏛️ Classement des universités** : Classement de l'université du premier cycle.
    - **🚀 Intérêt entrepreneurial** : Échelle de 1 à 10.
    - **🔗 Importance du réseautage** : Échelle de 1 à 10.
    - **💳 Source de financement du MBA** : Autofinancement, prêt, bourse...
    - **🏆 Rôle post-MBA souhaité** : Consultant, cadre, entrepreneur...
    - **📈 Salaire post-MBA attendu** : Salaire attendu après le MBA.
    - **🌍 Préférence de localisation** : Préférence géographique post-MBA.
    - **🎯 Raison du MBA** : Croissance de carrière, entrepreneuriat, compétences...
    - **🏫 Format MBA** : En ligne ou sur campus.
    """)


