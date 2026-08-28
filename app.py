import streamlit as st

st.set_page_config(page_title="PrediFoot AI", page_icon="⚽", layout="wide")

st.title("⚽ PrediFoot AI - Analyseur de Matchs")
st.write("Bienvenue sur votre application personnelle de prédiction footballistique.")

col1, col2 = st.columns(2)

with col1:
    equipe_dom = st.text_input("Équipe Domicile", "PSG")
    xg_dom = st.slider("xG moyen Domicile (5 derniers matchs)", 0.5, 3.5, 1.8)

with col2:
    equipe_ext = st.text_input("Équipe Extérieur", "Marseille")
    xg_ext = st.slider("xG moyen Extérieur (5 derniers matchs)", 0.5, 3.5, 1.2)

if st.button("Calculer les Probabilités", type="primary"):
    prob_dom = min(85, max(10, int((xg_dom / (xg_dom + xg_ext)) * 100)))
    prob_ext = min(85, max(10, int((xg_ext / (xg_dom + xg_ext)) * 100)))
    prob_nul = 100 - prob_dom - prob_ext

    st.subheader("📊 Résultats de la prédiction")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(f"Victoire {equipe_dom}", f"{prob_dom}%")
    col_b.metric("Match Nul", f"{prob_nul}%")
    col_c.metric(f"Victoire {equipe_ext}", f"{prob_ext}%")
    
    total_xg = xg_dom + xg_ext
    over_25 = "Oui" if total_xg > 2.5 else "Non"
    st.info(f"💡 **Plus de 2.5 buts (Over 2.5) :** {over_25} (xG total estimé : {total_xg:.2f})")
