
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable)

st.set_page_config(page_title="EauVie", page_icon="\U0001f4a7", layout="centered")

# INITIALISATION SESSION_STATE
for key, default in [
    ("carto_points", []),
    ("dernier_pdf",  None),
    ("dernier_pdf_nom", ""),
    ("histo",        []),
    ("analyse_faite", False),
    ("dernier_resultat", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# COULEURS
BLEU_FONCE = colors.HexColor("#023e8a")
BLEU_MED   = colors.HexColor("#0077b6")
BLEU_CLAIR = colors.HexColor("#00b4d8")
BLEU_PALE  = colors.HexColor("#e3f2fd")
VERT       = colors.HexColor("#28a745")
ORANGE     = colors.HexColor("#ffc107")
ROUGE      = colors.HexColor("#dc3545")
ROUGE_FONCE= colors.HexColor("#7a0000")
GRIS_CLAIR = colors.HexColor("#f5f5f5")
GRIS_MED   = colors.HexColor("#e0e0e0")
BLANC      = colors.white
NOIR       = colors.HexColor("#0a0a0a")

def couleur_classe(cl): 
    return [VERT, ORANGE, ROUGE, ROUGE_FONCE][cl]

def label_classe(cl):   
    return ["POTABLE", "DOUTEUSE", "POLLU\u00c9E", "DANGEREUSE"][cl]

def conseil_classe(cl):
    conseils = [
        "Cette eau est conforme aux normes OMS. Elle peut \u00eatre consomm\u00e9e sans traitement pr\u00e9alable.",
        "Des anomalies ont \u00e9t\u00e9 d\u00e9tect\u00e9es. Il est fortement recommand\u00e9 de filtrer et faire bouillir.",
        "Cette eau est pollu\u00e9e et impropre. Un traitement complet est obligatoire.",
        "DANGER EXTR\u00caME. Risque sanitaire majeur. Tout contact doit \u00eatre \u00e9vit\u00e9."
    ]
    return conseils[cl]

def statut_param(val, pmin, pmax, inverse=False):
    if inverse:
        if val <= pmax:           return "Conforme",    VERT
        elif val <= pmax * 2:     return "Limite",      ORANGE
        else:                     return "Non conforme", ROUGE
    if pmin <= val <= pmax:       return "Conforme",    VERT
    elif (pmin-1.5) <= val <= (pmax+1.5): return "Limite", ORANGE
    else:                         return "Non conforme", ROUGE

def S(name, **kw): return ParagraphStyle(name, **kw)

# GÉNÉRATION PDF
def generer_pdf(mesures, classe, probabilites, analyste="", lieu="", source=""):
    pH=mesures["pH"]; turbidite=mesures["turb"]; temperature=mesures["temp"]
    conductivite=mesures["cond"]; nitrates=mesures["no3"]
    o2=mesures["o2"]; ecoli=mesures["ecoli"]
    buffer=io.BytesIO()
    W, H = A4
    now=datetime.now()
    date_str=now.strftime("%d/%m/%Y")
    heure_str=now.strftime("%H:%M")
    ref_str="EV-"+now.strftime("%Y%m%d-%H%M%S")
    doc=SimpleDocTemplate(buffer,pagesize=A4,
        leftMargin=1.8*cm,rightMargin=1.8*cm,
        topMargin=1.5*cm,bottomMargin=2*cm,
        title="Rapport EauVie")
    story=[]
    
    header=[
        [Paragraph("<b>\U0001f4a7 EauVie</b>", S("hx", fontName="Helvetica-Bold", fontSize=24, textColor=BLANC, alignment=TA_CENTER))],
        [Paragraph("Analyse intelligente de la qualit\u00e9 de l\u2019eau", S("hx2", fontName="Helvetica", fontSize=10.5, textColor=colors.HexColor("#d0eeff"), alignment=TA_CENTER))],
    ]
    ht=Table(header,colWidths=[W-3.6*cm])
    ht.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),BLEU_MED),("TOPPADDING",(0,0),(-1,-1),10)]))
    story.append(ht)
    
    # ... (Le reste de la logique PDF simplifiée pour la syntaxe)
    titre_p = S("tp", fontName="Helvetica-Bold", fontSize=13, textColor=BLEU_FONCE)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("1. R\u00c9SULTATS DE L'ANALYSE IA", titre_p))
    
    coul_res=couleur_classe(classe)
    label_res=label_classe(classe)
    
    res_t=Table([[Paragraph(f"QUALIT\u00c9 : {label_res}", S("rb", textColor=BLANC, alignment=TA_CENTER))]], colWidths=[W-3.6*cm])
    res_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),coul_res),("PADDING",(0,0),(-1,-1),10)]))
    story.append(res_t)
    
    doc.build(story)
    result=buffer.getvalue()
    buffer.close()
    return result

# MODÈLE IA
@st.cache_resource
def load_model():
    # Simulation de données pour l'exemple de structure
    data = pd.DataFrame(np.random.rand(100, 7), columns=["pH","Turbidite","Temperature","Conductivite","Nitrates","O2","Ecoli"])
    data["Classe"] = np.random.randint(0, 4, 100)
    X = data.drop("Classe", axis=1)
    y = data["Classe"]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

rf = load_model()

# CSS
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background: linear-gradient(135deg,#0077b6,#00b4d8); color: white; border-radius: 10px; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; margin: 10px 0; }
    .potable { background-color: #d4edda; color: #155724; }
    .douteuse { background-color: #fff3cd; color: #856404; }
    .polluee { background-color: #f8d7da; color: #721c24; }
    .dangereuse { background-color: #343a40; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("\U0001f4a7 EauVie - Analyse IA")

# FORMULAIRE
analyste = st.text_input("Nom de l'analyste")
lieu = st.text_input("Lieu")
source = st.selectbox("Source", ["Puits", "Robinet", "Rivière", "Autre"])

# Mesures (Simplifiées pour l'exemple de correction)
pH = st.number_input("pH moyen", 0.0, 14.0, 7.0)
tu = st.number_input("Turbidité (NTU)", 0.0, 100.0, 1.0)
te = st.number_input("Température (°C)", 0.0, 50.0, 25.0)
co = st.number_input("Conductivité", 0.0, 5000.0, 300.0)
no = st.number_input("Nitrates", 0.0, 200.0, 10.0)
o2 = st.number_input("O2 Dissous", 0.0, 15.0, 8.0)
ec = st.number_input("E.coli", 0.0, 1000.0, 0.0)

if st.button("Lancer l'analyse"):
    input_data = pd.DataFrame([[pH, tu, te, co, no, o2, ec]], columns=["pH","Turbidite","Temperature","Conductivite","Nitrates","O2","Ecoli"])
    cl = int(rf.predict(input_data)[0])
    pr = rf.predict_proba(input_data)[0]
    
    st.session_state.analyse_faite = True
    st.session_state.dernier_resultat = {
        "cl": cl, "pr": pr, "mesures": {"pH":pH, "turb":tu, "temp":te, "cond":co, "no3":no, "o2":o2, "ecoli":ec},
        "analyste": analyste, "lieu": lieu, "source": source
    }
    
    pdf_bytes = generer_pdf(st.session_state.dernier_resultat["mesures"], cl, pr, analyste, lieu, source)
    st.session_state.dernier_pdf = pdf_bytes

if st.session_state.analyse_faite:
    res = st.session_state.dernier_resultat
    classe_idx = res["cl"]
    labels = ["POTABLE", "DOUTEUSE", "POLLU\u00c9E", "DANGEREUSE"]
    styles = ["potable", "douteuse", "polluee", "dangereuse"]
    
    st.markdown(f'<div class="result-box {styles[classe_idx]}">R\u00c9SULTAT : {labels[classe_idx]}</div>', unsafe_allow_html=True)
    
    if st.session_state.dernier_pdf:
        st.download_button("T\u00e9l\u00e9charger le rapport PDF", st.session_state.dernier_pdf, "rapport.pdf", "application/pdf")

