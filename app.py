import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Indice de performance d'un étudiant", page_icon="🎓", layout="centered")

# ---------- Les variables, dans l'ordre utilisé pour l'entrainement dans le notebook ----------
FEATURES = ["Hours Studied", "Previous Scores", "Extracurricular Activities",
            "Sleep Hours", "Sample Question Papers Practiced"]
CIBLE = "Performance Index"

# ---------- Chargement des fichiers produits par le notebook (cherchés à côté de app.py) ----------
DOSSIER = Path(__file__).parent

@st.cache_resource
def charger():
    encoders = jb.load(DOSSIER / "encoders.joblib")   # encodeur de la variable catégorielle
    scaler = jb.load(DOSSIER / "scaler.joblib")       # RobustScaler utilisé dans le notebook
    model = jb.load(DOSSIER / "best_model.joblib")    # modèle le plus performant
    return encoders, scaler, model

encoders, scaler, model = charger()

# ---------- Prédiction simple ----------
def Pred_func(Hours_Studied, Previous_Scores, Extracurricular_Activities, Sleep_Hours, Sample_Papers):
    # Encoder la variable catégorielle avec le même encodeur que dans le notebook
    Extracurricular_Activities = encoders[0].transform([Extracurricular_Activities])[0]
    # Même ordre que x dans le notebook
    x_new = np.array([Hours_Studied, Previous_Scores, Extracurricular_Activities, Sleep_Hours, Sample_Papers],
                     dtype=float).reshape(1, -1)
    # Dans le notebook, x était normalisé avant d'être découpé : on applique le même scaler
    x_new = scaler.transform(x_new)
    return round(float(model.predict(x_new)[0]), 2)

# ---------- Prédiction multiple ----------
def Pred_func_csv(file):
    df = pd.read_csv(file)
    manquantes = [c for c in FEATURES if c not in df.columns]
    if manquantes:
        raise KeyError("Colonnes manquantes : " + ", ".join(manquantes))
    df[CIBLE + " (prédit)"] = [Pred_func(*ligne) for ligne in df[FEATURES].values]
    return df

# ---------- Interface ----------
st.title("🎓 Indice de performance d'un étudiant")
st.write("Ce modèle prédit l'indice de performance d'un étudiant à partir du nombre d'heures étudiées, "
         "des notes précédentes, des activités extra-scolaires, des heures de sommeil et du nombre "
         "d'annales pratiquées.")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

with onglet1:
    with st.form("formulaire"):
        col1, col2 = st.columns(2)
        with col1:
            Hours_Studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5, step=1)
            Previous_Scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70, step=1)
            Sleep_Hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7, step=1)
        with col2:
            Sample_Papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=50, value=5, step=1)
            Extracurricular_Activities = st.radio("Extracurricular Activities", list(encoders[0].classes_))
        soumettre = st.form_submit_button("Prédire", type="primary")
    if soumettre:
        indice = Pred_func(Hours_Studied, Previous_Scores, Extracurricular_Activities, Sleep_Hours, Sample_Papers)
        st.success(f"**{CIBLE} :** {indice}")

with onglet2:
    st.write("Importer un fichier csv avec les colonnes : " + ", ".join(FEATURES))
    fichier = st.file_uploader("Importer un fichier csv", type=["csv"])
    if fichier is not None:
        try:
            df_resultat = Pred_func_csv(fichier)
            st.dataframe(df_resultat)
            st.download_button("Télécharger les prédictions", df_resultat.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")
        except KeyError as e:
            st.error(str(e))
        except ValueError as e:
            st.error(f"Une valeur du fichier n'existe pas dans les données d'entrainement : {e}")
