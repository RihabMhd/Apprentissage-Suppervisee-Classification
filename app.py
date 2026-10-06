import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Segmentation Client",page_icon="🛍️")

MODEL_PATH=Path(__file__).parent /"models"/"rf_segmentation.joblib"

if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle=load_bundle()

model=bundle["model"]
names=bundle["cluster_names"]
features=bundle["features"]

st.title("Segmentation Client 🛍️")
st.text("Saisissez les informations RFM d'un client pour connaître son segment")

with st.form("rfm_form"):
    recency = st.number_input("Récence (jours)", min_value=1, max_value=400, value=51, step=1,
                              help="Nombre de jours depuis le dernier achat")
    frequency = st.number_input("Fréquence (nombre de commandes)", min_value=1, max_value=250, value=2, step=1,
                                help="Nombre de commandes passées par le client")
    monetary = st.number_input("Montant total dépensé (£)", min_value=0.0, max_value=300000.0, value=661.0, step=10.0,
                               help="Total dépensé par le client depuis le début")
    submitted = st.form_submit_button("Prédire le segment")

if submitted:
    client = pd.DataFrame([{"Recency": recency, "Frequency": frequency, "Monetary": monetary}])
    cluster = int(model.predict(client[features])[0])
    segment = names[cluster]

    st.success(f"Segment : {segment}")
    st.write(f"Cluster prédit : {cluster}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Récence (jours)", recency)
    col2.metric("Fréquence", frequency)
    col3.metric("Montant (DH)", f"{monetary:,.2f}")

    st.caption("Le segment est attribué selon les règles de segmentation apprises à partir du K-Means ; il ne prédit pas le comportement futur du client.")