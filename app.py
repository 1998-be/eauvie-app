import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='EauVie', page_icon='💧', layout='centered')

@st.cache_resource
def load_model():
    data = pd.DataFrame([
        {'pH':7.2,'Turbidite':0.8,'Absorbance':0.08,'O2':7.5,'Classe':0},
        {'pH':7.5,'Turbidite':1.2,'Absorbance':0.10,'O2':7.2,'Classe':0},
        {'pH':7.0,'Turbidite':0.3,'Absorbance':0.05,'O2':8.0,'Classe':0},
        {'pH':6.8,'Turbidite':2.0,'Absorbance':0.12,'O2':6.8,'Classe':0},
        {'pH':7.3,'Turbidite':0.5,'Absorbance':0.06,'O2':7.8,'Classe':0},
        {'pH':7.1,'Turbidite':0.6,'Absorbance':0.07,'O2':6.5,'Classe':0},
        {'pH':7.4,'Turbidite':1.0,'Absorbance':0.09,'O2':7.0,'Classe':0},
        {'pH':6.6,'Turbidite':1.5,'Absorbance':0.11,'O2':6.9,'Classe':0},
        {'pH':7.6,'Turbidite':0.9,'Absorbance':0.08,'O2':7.4,'Classe':0},
        {'pH':7.2,'Turbidite':0.7,'Absorbance':0.06,'O2':7.6,'Classe':0},
        {'pH':7.8,'Turbidite':1.8,'Absorbance':0.13,'O2':6.6,'Classe':0},
        {'pH':6.9,'Turbidite':2.5,'Absorbance':0.15,'O2':6.5,'Classe':0},
        {'pH':6.7,'Turbidite':0.2,'Absorbance':0.04,'O2':7.9,'Classe':0},
        {'pH':7.0,'Turbidite':0.4,'Absorbance':0.05,'O2':8.2,'Classe':0},
        {'pH':7.3,'Turbidite':1.1,'Absorbance':0.09,'O2':7.1,'Classe':0},
        {'pH':6.3,'Turbidite':5.5,'Absorbance':0.28,'O2':5.8,'Classe':1},
        {'pH':5.8,'Turbidite':4.5,'Absorbance':0.22,'O2':5.5,'Classe':1},
        {'pH':6.2,'Turbidite':6.0,'Absorbance':0.30,'O2':5.2,'Classe':1},
        {'pH':7.8,'Turbidite':7.0,'Absorbance':0.35,'O2':5.0,'Classe':1},
        {'pH':7.9,'Turbidite':8.0,'Absorbance':0.40,'O2':4.8,'Classe':1},
        {'pH':6.4,'Turbidite':9.0,'Absorbance':0.45,'O2':5.5,'Classe':1},
        {'pH':6.5,'Turbidite':6.5,'Absorbance':0.32,'O2':5.0,'Classe':1},
        {'pH':8.6,'Turbidite':5.8,'Absorbance':0.29,'O2':5.3,'Classe':1},
        {'pH':5.5,'Turbidite':4.8,'Absorbance':0.25,'O2':5.6,'Classe':1},
        {'pH':6.1,'Turbidite':7.5,'Absorbance':0.38,'O2':4.9,'Classe':1},
        {'pH':8.0,'Turbidite':5.0,'Absorbance':0.27,'O2':5.7,'Classe':1},
        {'pH':6.0,'Turbidite':25.0,'Absorbance':0.75,'O2':3.8,'Classe':2},
        {'pH':5.8,'Turbidite':35.0,'Absorbance':0.90,'O2':3.2,'Classe':2},
        {'pH':5.5,'Turbidite':20.0,'Absorbance':0.70,'O2':3.5,'Classe':2},
        {'pH':6.3,'Turbidite':40.0,'Absorbance':1.10,'O2':3.0,'Classe':2},
        {'pH':5.9,'Turbidite':30.0,'Absorbance':0.85,'O2':3.4,'Classe':2},
        {'pH':9.2,'Turbidite':28.0,'Absorbance':0.95,'O2':2.8,'Classe':2},
        {'pH':5.6,'Turbidite':50.0,'Absorbance':1.40,'O2':2.5,'Classe':2},
        {'pH':6.2,'Turbidite':18.0,'Absorbance':0.65,'O2':3.6,'Classe':2},
        {'pH':6.5,'Turbidite':45.0,'Absorbance':1.20,'O2':2.9,'Classe':2},
        {'pH':5.7,'Turbidite':55.0,'Absorbance':1.50,'O2':2.2,'Classe':2},
        {'pH':4.8,'Turbidite':22.0,'Absorbance':0.80,'O2':3.7,'Classe':2},
        {'pH':4.2,'Turbidite':90.0,'Absorbance':2.50,'O2':1.0,'Classe':3},
        {'pH':5.0,'Turbidite':95.0,'Absorbance':2.80,'O2':0.8,'Classe':3},
        {'pH':4.5,'Turbidite':80.0,'Absorbance':2.20,'O2':0.5,'Classe':3},
        {'pH':3.8,'Turbidite':100.0,'Absorbance':3.00,'O2':0.3,'Classe':3},
        {'pH':8.8,'Turbidite':85.0,'Absorbance':2.40,'O2':0.6,'Classe':3},
        {'pH':4.0,'Turbidite':70.0,'Absorbance':2.00,'O2':1.2,'Classe':3},
        {'pH':4.5,'Turbidite':65.0,'Absorbance':1.90,'O2':0.9,'Classe':3},
        {'pH':4.1,'Turbidite':75.0,'Absorbance':2.10,'O2':0.7,'Classe':3},
        {'pH':4.8,'Turbidite':88.0,'Absorbance':2.60,'O2':0.4,'Classe':3},
        {'pH':3.5,'Turbidite':98.0,'Absorbance':2.90,'O2':0.2,'Classe':3},
        {'pH':4.3,'Turbidite':92.0,'Absorbance':2.70,'O2':0.6,'Classe':3},
        {'pH':3.9,'Turbidite':97.0,'Absorbance':2.95,'O2':0.1,'Classe':3},
    ])
    X = data[['pH','Turbidite','Absorbance','O2']]
    y = data['Classe']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    rf = RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,class_weight='balanced')
    rf.fit(X_train,y_train)
    return rf

rf = load_model()

st.markdown('''<style>
html,body,[class*='css']{color:#0a0a0a !important;}
.main{background:linear-gradient(160deg,#dff3fb 0%,#e8f4fd 100%);}
.block-container{background:rgba(255,255,255,0.97);border-radius:18px;padding:2rem;box-shadow:0 4px 32px rgba(0,119,182,0.12);}
.stButton>button{background:linear-gradient(135deg,#0077b6,#00b4d8);color:white !important;font-size:17px;border-radius:14px;padding:14px 30px;width:100%;border:none;font-weight:700;letter-spacing:1px;box-shadow:0 4px 15px rgba(0,119,182,0.3);}
.result-box{padding:24px;border-radius:16px;text-align:center;font-size:24px;font-weight:800;margin:18px 0;box-shadow:0 4px 20px rgba(0,0,0,0.1);}
.potable{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);color:#0a4a0a !important;border:3px solid #28a745;}
.douteuse{background:linear-gradient(135deg,#fff3cd,#ffe082);color:#4a3000 !important;border:3px solid #ffc107;}
.polluee{background:linear-gradient(135deg,#ffd5d5,#ffab91);color:#5a0000 !important;border:3px solid #dc3545;}
.dangereuse{background:linear-gradient(135deg,#2d0000,#1a0000);color:#ff6666 !important;border:3px solid #ff0000;}
.pcard{background:linear-gradient(135deg,#f0f8ff,#e3f2fd);border-left:5px solid #0077b6;border-radius:12px;padding:14px 16px;margin-bottom:12px;box-shadow:0 2px 8px rgba(0,119,182,0.10);}
.plabel{font-weight:800;color:#023e8a !important;font-size:15px;margin-bottom:6px;display:block;}
.ptext{color:#0a0a0a !important;font-size:13px;font-weight:500;line-height:1.6;display:block;margin-bottom:4px;}
.pnorm{font-size:12px;color:#023e8a !important;margin-top:7px;font-weight:700;background:rgba(0,119,182,0.10);padding:4px 10px;border-radius:6px;display:inline-block;}
.proto-box{background:linear-gradient(135deg,#f0fff4,#e8f8e8);border-left:5px solid #1b5e20;border-radius:12px;padding:14px 18px;margin-bottom:14px;}
.proto-title{font-weight:800;color:#1b5e20 !important;font-size:15px;margin-bottom:10px;display:block;}
.proto-item{color:#0a2a0a !important;font-size:13px;font-weight:500;padding:5px 0;border-bottom:1px solid rgba(27,94,32,0.12);line-height:1.5;display:block;}
.header-box{background:linear-gradient(135deg,#023e8a,#0077b6,#00b4d8);border-radius:16px;padding:22px 16px;text-align:center;margin-bottom:22px;box-shadow:0 4px 24px rgba(0,119,182,0.35);}
.header-title{color:#ffffff !important;font-size:30px;font-weight:800;letter-spacing:2px;}
.header-sub{color:#d0eeff !important;font-size:13px;margin-top:8px;line-height:1.6;}
.header-author{color:#a8d8ff !important;font-size:12px;margin-top:6px;font-style:italic;}
.section-title{color:#023e8a !important;font-size:17px;font-weight:800;border-bottom:2px solid #00b4d8;padding-bottom:7px;margin:18px 0 12px 0;display:block;}
.conseil-box{background:linear-gradient(135deg,#e3f2fd,#e0f7fa);border-left:5px solid #0077b6;border-radius:12px;padding:14px 18px;margin-top:10px;}
.conseil-title{font-weight:800;color:#023e8a !important;font-size:14px;margin-bottom:8px;display:block;}
.conseil-item{color:#0a0a0a !important;font-size:13px;padding:4px 0;border-bottom:1px solid rgba(0,119,182,0.10);display:block;line-height:1.5;}
p,span,div,label{color:#0a0a0a !important;}
</style>''', unsafe_allow_html=True)

st.markdown('<div class=\'header-box\'><div class=\'header-title\'>💧 EauVie</div><div class=\'header-sub\'>Analyse intelligente de la qualité de l’eau — Normes OMS<br>afin de garantir une consommation rassurante et bénéfique.</div><div class=\'header-author\'>Proposée par Charles MEDEZOUNDJI</div></div>', unsafe_allow_html=True)

with st.expander('📋 Normes OMS de référence'):
    st.markdown('| Paramètre | ✅ Potable | ⚠️ Douteuse | ❌ Polluée | ☠️ Dangereuse |')
    st.markdown('|---|---|---|---|---|')
    st.markdown('| pH | 6,5 à 8,5 | 5,5 à 9,0 | 4,5 à 5,5 | inférieur à 4,5 |')
    st.markdown('| Turbidité NTU | inférieur à 5 | 5 à 10 | 10 à 50 | supérieur à 50 |')
    st.markdown('| Absorbance | inférieur à 0,2 | 0,2 à 0,5 | 0,5 à 1,5 | supérieur à 1,5 |')
    st.markdown('| O₂ mg/L | supérieur à 6 | 4 à 6 | 2 à 4 | inférieur à 2 |')

with st.expander('🔬 Protocole de mesure'):
    st.markdown('<div class=\'proto-box\'><span class=\'proto-title\'>🧪 pH — Potentiomètre</span><span class=\'proto-item\'>🔧 Outil : pH-mètre numérique ou bandelettes de pH</span><span class=\'proto-item\'>1. Étalonner avec solutions tampon pH 4, 7 et 10</span><span class=\'proto-item\'>2. Plonger l’électrode dans l’échantillon</span><span class=\'proto-item\'>3. Attendre la stabilisation (30 secondes)</span><span class=\'proto-item\'>4. Lire et noter la valeur affichée</span><span class=\'proto-item\'>5. Rincer l’électrode à l’eau distillée après usage</span></div>', unsafe_allow_html=True)
    st.markdown('<div class=\'proto-box\'><span class=\'proto-title\'>🌊 Turbidité — Turbidimètre (NTU)</span><span class=\'proto-item\'>🔧 Outil : Turbidimètre numérique</span><span class=\'proto-item\'>1. Remplir le tube avec l’échantillon d’eau</span><span class=\'proto-item\'>2. Essuyer le tube pour éliminer les traces</span><span class=\'proto-item\'>3. Insérer dans le turbidimètre et fermer</span><span class=\'proto-item\'>4. Lire la valeur en NTU affichée</span><span class=\'proto-item\'>5. Répéter 3 fois et calculer la moyenne</span></div>', unsafe_allow_html=True)
    st.markdown('<div class=\'proto-box\'><span class=\'proto-title\'>🔵 Absorbance — Spectrophotomètre UV (254 nm)</span><span class=\'proto-item\'>🔧 Outil : Spectrophotomètre UV-Visible réglé à 254 nm</span><span class=\'proto-item\'>1. Étalonner avec eau distillée comme blanc</span><span class=\'proto-item\'>2. Filtrer l’échantillon sur membrane 0,45 µm</span><span class=\'proto-item\'>3. Remplir la cuvette avec l’échantillon filtré</span><span class=\'proto-item\'>4. Lancer la mesure et lire la valeur</span><span class=\'proto-item\'>5. Nettoyer la cuvette après chaque mesure</span></div>', unsafe_allow_html=True)
    st.markdown('<div class=\'proto-box\'><span class=\'proto-title\'>💨 Oxygène dissous — Oxymètre électronique</span><span class=\'proto-item\'>🔧 Outil : Oxymètre portable avec sonde à membrane</span><span class=\'proto-item\'>1. Étalonner dans l’air saturé en humidité (10 min)</span><span class=\'proto-item\'>2. Rincer la sonde à l’eau distillée</span><span class=\'proto-item\'>3. Plonger dans l’échantillon sans bulles d’air</span><span class=\'proto-item\'>4. Agiter doucement et attendre 2 minutes</span><span class=\'proto-item\'>5. Lire la valeur en mg/L affichée</span></div>', unsafe_allow_html=True)

st.markdown('<span class=\'section-title\'>🔬 Insérez les mesures de votre échantillon</span>', unsafe_allow_html=True)
st.markdown('<div class=\'pcard\'><span class=\'plabel\'>🧪 pH — Potentiel Hydrogène</span><span class=\'ptext\'>Mesure l’acidité ou la basicité de l’eau. pH bas : risque de métaux toxiques. pH élevé : contamination minérale ou chimique.</span><span class=\'pnorm\'>Norme OMS : 6,5 à 8,5</span></div>', unsafe_allow_html=True)
pH = st.number_input('pH', 0.0, 14.0, 7.0, 0.1)
st.markdown('<div class=\'pcard\'><span class=\'plabel\'>🌊 Turbidité (NTU) — Trouble de l’eau</span><span class=\'ptext\'>Particules en suspension : argile, bactéries, matières organiques. Eau trouble : agents pathogènes possibles.</span><span class=\'pnorm\'>Norme OMS : inférieur à 5 NTU</span></div>', unsafe_allow_html=True)
tu = st.number_input('Turbidité (NTU)', 0.0, 200.0, 2.0, 0.5)
st.markdown('<div class=\'pcard\'><span class=\'plabel\'>🔵 Absorbance — Matières organiques dissoutes</span><span class=\'ptext\'>Capacité de l’eau à absorber la lumière UV. Valeur élevée : polluants organiques ou chimiques dissous.</span><span class=\'pnorm\'>Seuil potable : inférieur à 0,2</span></div>', unsafe_allow_html=True)
ab = st.number_input('Absorbance', 0.0, 5.0, 0.1, 0.01)
st.markdown('<div class=\'pcard\'><span class=\'plabel\'>💨 Oxygène dissous (mg/L) — Vitalité de l’eau</span><span class=\'ptext\'>Taux faible : pollution organique intense, putréfaction. Indicateur clé de la santé de l’eau.</span><span class=\'pnorm\'>Norme : supérieur à 6 mg/L. Inférieur à 2 mg/L : eau dangereuse</span></div>', unsafe_allow_html=True)
o2 = st.number_input('Oxygène dissous (mg/L)', 0.0, 14.0, 7.0, 0.1)
st.markdown('---')
MP = {
    0: ('💧 POTABLE', 'potable', 'Eau conforme aux normes OMS. Consommation possible sans risque.'),
    1: ('⚠️ DOUTEUSE', 'douteuse', 'Anomalies détectées. Filtrez et faites bouillir avant consommation.'),
    2: ('❌ POLLUÉE', 'polluee', 'Eau polluée. Ne pas consommer. Traitement obligatoire.'),
    3: ('☠️ DANGEREUSE', 'dangereuse', 'DANGER EXTRÊME. Tout contact à éviter. Risque sanitaire majeur.'),
}
if st.button('🔍 Analyser la qualité de l’eau'):
    dfm = pd.DataFrame({'pH':[pH],'Turbidite':[tu],'Absorbance':[ab],'O2':[o2]})
    cl = rf.predict(dfm)[0]
    pr = rf.predict_proba(dfm)[0]
    lb,cs,co = MP[cl]
    conf = str(round(pr[cl]*100,1))
    st.markdown('<div class=\'result-box '+cs+'\'>'+lb+'<br><span style=\'font-size:14px;font-weight:600;\'>Confiance du modèle : '+conf+' %</span></div>', unsafe_allow_html=True)
    st.markdown('**💡 Conseil :** '+co)
    if cl in [1,2,3]:
        with st.expander('🛠️ Comment purifier cette eau ?'):
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>🔥 1. Ébullition</span><span class=\'conseil-item\'>Porter l’eau à ébullition pendant au moins 5 minutes.</span><span class=\'conseil-item\'>Laisser refroidir dans un récipient propre et couvert.</span><span class=\'conseil-item\'>Efficace contre bactéries, virus et parasites.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>🧴 2. Filtration sur sable et gravier</span><span class=\'conseil-item\'>Couches : gravier grossier, gravier fin, sable grossier, sable fin, charbon de bois.</span><span class=\'conseil-item\'>Verser l’eau par le dessus, récupérer par le bas.</span><span class=\'conseil-item\'>Compléter obligatoirement avec l’ébullition.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>☀️ 3. Désinfection solaire SODIS</span><span class=\'conseil-item\'>Remplir des bouteilles transparentes avec l’eau filtrée.</span><span class=\'conseil-item\'>Exposer 6 heures au soleil (ciel clair) ou 2 jours (nuageux).</span><span class=\'conseil-item\'>Méthode gratuite et prouvée par l’OMS, idéale pour l’Afrique de l’Ouest.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>🧪 4. Chloration</span><span class=\'conseil-item\'>2 gouttes d’eau de Javel à 5 % par litre d’eau trouble, 1 goutte par litre d’eau claire.</span><span class=\'conseil-item\'>Mélanger et attendre 30 minutes avant de consommer.</span><span class=\'conseil-item\'>Efficace contre bactéries et virus.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>🌱 5. Graines de Moringa</span><span class=\'conseil-item\'>Broyer 2 à 3 graines sèches en poudre fine.</span><span class=\'conseil-item\'>Ajouter à 1 litre d’eau turbide, agiter 1 min puis 5 min lentement.</span><span class=\'conseil-item\'>Décanter 1 heure puis compléter par ébullition ou chloration.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>💧 6. Filtre à membrane</span><span class=\'conseil-item\'>Filtre céramique ou à fibres creuses à 0,2 micron.</span><span class=\'conseil-item\'>Retient bactéries et parasites. Compléter avec chloration contre les virus.</span><span class=\'conseil-item\'>Filtres LifeStraw ou Sawyer disponibles dans les ONG locales.</span></div>', unsafe_allow_html=True)
    prd = pd.DataFrame({'Classe':['Potable','Douteuse','Polluée','Dangereuse'],'Probabilité (%)':[round(p*100,1) for p in pr]})
    st.bar_chart(prd.set_index('Classe'))
    if 'histo' not in st.session_state: st.session_state.histo=[]
    st.session_state.histo.append({'Heure':datetime.now().strftime('%H:%M:%S'),'pH':pH,'Turbidité':tu,'Absorbance':ab,'O₂':o2,'Résultat':lb})
if 'histo' in st.session_state and len(st.session_state.histo)>0:
    st.markdown('---')
    st.markdown('<span class=\'section-title\'>🕔 Historique des analyses</span>', unsafe_allow_html=True)
    hdf = pd.DataFrame(st.session_state.histo)
    st.dataframe(hdf,use_container_width=True)
    st.download_button('⬇️ Télécharger le CSV',hdf.to_csv(index=False).encode('utf-8'),'historique_eauvie.csv','text/csv')
st.markdown('---')
st.markdown('<div style=\'text-align:center;color:#023e8a !important;font-size:12px;padding:10px;font-weight:600;\'>💧 EauVie — Random Forest — Normes OMS — Charles MEDEZOUNDJI</div>', unsafe_allow_html=True)
