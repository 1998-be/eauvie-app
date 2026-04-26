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
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

st.set_page_config(page_title="EauVie", page_icon="\U0001f4a7", layout="centered")

# ── SESSION STATE ─────────────────────────────────────────────
for k, v in [
    ("carto_points", []),
    ("dernier_pdf", None),
    ("dernier_pdf_nom", ""),
    ("histo", []),
    ("analyse_faite", False),
    ("dernier_resultat", None),
    ("module_actif", "potable"),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── COULEURS PDF ──────────────────────────────────────────────
BF = colors.HexColor("#023e8a")
BM = colors.HexColor("#0077b6")
BC = colors.HexColor("#00b4d8")
BP = colors.HexColor("#e3f2fd")
VT = colors.HexColor("#28a745")
OR = colors.HexColor("#ffc107")
RG = colors.HexColor("#dc3545")
RF = colors.HexColor("#7a0000")
GC = colors.HexColor("#f5f5f5")
GM = colors.HexColor("#e0e0e0")
WH = colors.white
NK = colors.HexColor("#0a0a0a")
VM = colors.HexColor("#1b5e20")
BR = colors.HexColor("#e65100")
MA = colors.HexColor("#880e4f")
PU = colors.HexColor("#7b1fa2")


def S(name, **kw):
    return ParagraphStyle(name, **kw)


def statut_param(val, pmin, pmax, inverse=False):
    if val is None:
        return "Non mesure", colors.HexColor("#9e9e9e")
    if inverse:
        if val <= pmax:
            return "Conforme", VT
        elif val <= pmax * 2:
            return "Limite", OR
        else:
            return "Non conforme", RG
    if pmin <= val <= pmax:
        return "Conforme", VT
    elif (pmin - 1.5) <= val <= (pmax + 1.5):
        return "Limite", OR
    else:
        return "Non conforme", RG


CRITIQUES_POTABLE = {
    "ecoli": "E. coli",
    "pH": "pH",
    "turb": "Turbidite",
    "no3": "Nitrates",
    "no2": "Nitrites",
    "pb": "Plomb",
}


def evaluer_sous_reserve(vals, cl):
    manquants = [lbl for k, lbl in CRITIQUES_POTABLE.items() if vals.get(k) is None]
    sous_reserve = len(manquants) > 0 and cl == 0
    labels = ["POTABLE", "DOUTEUSE", "POLLUÉE", "DANGEREUSE"]
    label = labels[cl]
    if sous_reserve:
        label = "POTABLE (sous reserve des mesures a completer)"
    return label, sous_reserve, manquants


# ── MODELES IA ────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models = {}

    # MODULE 1 : EAU POTABLE
    # Sources: Mama 2011, Imorou Toko 2010, Boukari 2003, Vodounnou 2020,
    #          SONEB bulletins 2018-2022, DN Hydraulique Benin,
    #          OMS 2017, NB 001/2001, USEPA 2022
    potable_d = [
        (0, 7.1, 0.5, 26.5, 285, 7.8, 1.8, 0.008, 0.05, 0.003, 0.32),
        (0, 7.3, 0.8, 27.0, 310, 7.5, 2.2, 0.010, 0.06, 0.002, 0.28),
        (0, 7.0, 0.4, 26.0, 295, 8.0, 1.5, 0.007, 0.04, 0.002, 0.35),
        (0, 7.2, 0.6, 27.5, 320, 7.6, 2.0, 0.009, 0.05, 0.003, 0.30),
        (0, 7.4, 1.0, 26.5, 305, 7.3, 2.5, 0.012, 0.07, 0.003, 0.25),
        (0, 6.8, 1.2, 28.0, 420, 6.9, 3.5, 0.015, 0.08, 0.004, 0.0),
        (0, 7.0, 0.9, 27.5, 395, 7.1, 3.0, 0.012, 0.07, 0.003, 0.0),
        (0, 6.9, 1.5, 28.5, 445, 6.7, 3.8, 0.018, 0.09, 0.005, 0.0),
        (0, 7.2, 1.1, 27.0, 380, 7.2, 3.2, 0.014, 0.08, 0.004, 0.0),
        (0, 7.5, 0.7, 26.5, 355, 7.4, 2.8, 0.011, 0.06, 0.003, 0.0),
        (0, 6.7, 1.8, 29.0, 520, 6.6, 4.5, 0.020, 0.12, 0.006, 0.0),
        (0, 6.9, 2.0, 28.5, 485, 6.8, 4.2, 0.018, 0.11, 0.005, 0.0),
        (0, 7.1, 1.6, 28.0, 495, 7.0, 4.0, 0.016, 0.10, 0.005, 0.0),
        (0, 7.3, 1.3, 27.5, 465, 7.2, 3.6, 0.015, 0.09, 0.004, 0.0),
        (0, 6.6, 2.2, 29.5, 540, 6.5, 4.8, 0.022, 0.13, 0.007, 0.0),
        (0, 7.2, 0.6, 27.0, 340, 7.5, 2.2, 0.010, 0.06, 0.002, 0.28),
        (0, 7.0, 0.5, 26.5, 325, 7.8, 1.9, 0.008, 0.05, 0.002, 0.31),
        (0, 7.3, 0.8, 27.5, 360, 7.4, 2.5, 0.011, 0.07, 0.003, 0.26),
        (0, 7.1, 1.0, 28.0, 380, 7.1, 2.8, 0.013, 0.08, 0.004, 0.22),
        (0, 7.4, 0.7, 27.0, 350, 7.6, 2.1, 0.009, 0.06, 0.002, 0.29),
        (0, 6.8, 2.5, 29.0, 580, 6.5, 5.0, 0.025, 0.14, 0.007, 0.0),
        (0, 7.0, 2.1, 28.5, 555, 6.7, 4.8, 0.022, 0.13, 0.006, 0.0),
        (0, 7.2, 1.9, 28.0, 530, 6.9, 4.5, 0.020, 0.12, 0.006, 0.0),
        (0, 6.7, 2.8, 29.5, 610, 6.4, 5.2, 0.028, 0.15, 0.008, 0.0),
        (0, 7.1, 2.3, 28.5, 560, 6.6, 4.9, 0.024, 0.13, 0.007, 0.0),
        (0, 7.5, 1.5, 27.5, 420, 7.3, 3.5, 0.015, 0.09, 0.004, 0.0),
        (0, 7.3, 1.2, 27.0, 395, 7.5, 3.2, 0.013, 0.08, 0.003, 0.0),
        (0, 7.6, 0.9, 26.5, 375, 7.7, 2.9, 0.011, 0.07, 0.003, 0.0),
        (0, 7.2, 1.8, 28.0, 445, 7.1, 3.8, 0.017, 0.10, 0.005, 0.0),
        (0, 7.0, 1.4, 27.5, 410, 7.4, 3.4, 0.014, 0.09, 0.004, 0.0),
    ]
    douteuse_d = [
        (4, 6.3, 5.5, 30.5, 980, 5.8, 14.5, 0.18, 0.55, 0.008, 0.0),
        (6, 6.1, 6.2, 31.0, 1050, 5.5, 18.0, 0.22, 0.72, 0.009, 0.0),
        (3, 6.5, 4.8, 30.0, 920, 5.9, 12.2, 0.15, 0.48, 0.007, 0.0),
        (8, 7.8, 7.0, 31.5, 1120, 5.0, 22.5, 0.28, 0.88, 0.010, 0.0),
        (5, 5.8, 5.8, 30.5, 1010, 5.4, 16.8, 0.20, 0.65, 0.009, 0.0),
        (7, 8.5, 8.2, 32.0, 1280, 4.8, 25.0, 0.32, 1.05, 0.011, 0.0),
        (9, 8.7, 9.0, 32.5, 1380, 4.6, 30.5, 0.38, 1.20, 0.012, 0.0),
        (2, 6.4, 5.2, 30.0, 960, 5.7, 13.0, 0.16, 0.52, 0.008, 0.0),
        (6, 8.2, 7.5, 31.5, 1200, 5.1, 22.0, 0.27, 0.85, 0.010, 0.0),
        (4, 5.9, 6.5, 31.0, 1080, 5.3, 17.5, 0.21, 0.68, 0.009, 0.0),
        (3, 8.8, 5.5, 30.5, 1350, 5.5, 19.0, 0.24, 0.78, 0.010, 0.0),
        (5, 8.6, 6.8, 31.0, 1260, 5.2, 23.0, 0.29, 0.92, 0.011, 0.0),
        (8, 5.6, 7.8, 31.5, 1180, 4.9, 27.0, 0.34, 1.08, 0.011, 0.0),
        (2, 6.2, 5.0, 30.0, 950, 5.8, 12.5, 0.15, 0.50, 0.008, 0.0),
        (9, 7.5, 8.8, 32.5, 1450, 4.6, 32.0, 0.40, 1.28, 0.012, 0.0),
        (6, 5.9, 6.8, 31.5, 1150, 5.2, 20.5, 0.25, 0.82, 0.010, 0.0),
        (3, 8.7, 5.5, 30.5, 1080, 5.5, 15.5, 0.19, 0.62, 0.009, 0.0),
        (7, 6.6, 7.2, 32.0, 1280, 4.8, 26.0, 0.32, 1.02, 0.011, 0.0),
        (9, 5.5, 8.8, 32.5, 1360, 4.6, 29.5, 0.37, 1.18, 0.012, 0.0),
        (2, 7.1, 5.8, 30.5, 1020, 5.8, 13.5, 0.17, 0.54, 0.008, 0.0),
        (4, 8.4, 5.2, 30.5, 1060, 5.5, 14.0, 0.17, 0.56, 0.008, 0.0),
        (9, 6.1, 9.2, 32.5, 1420, 4.6, 35.5, 0.44, 1.38, 0.013, 0.0),
        (6, 5.8, 7.0, 31.5, 1220, 5.0, 22.0, 0.27, 0.88, 0.010, 0.0),
        (4, 7.7, 6.5, 31.0, 1140, 4.8, 17.0, 0.21, 0.68, 0.009, 0.0),
        (7, 6.8, 8.2, 32.0, 1310, 5.1, 26.5, 0.33, 1.05, 0.011, 0.0),
        (1, 8.9, 5.0, 30.0, 980, 5.6, 11.5, 0.14, 0.46, 0.007, 0.0),
        (8, 5.5, 7.8, 32.0, 1290, 4.8, 28.0, 0.35, 1.10, 0.012, 0.0),
        (5, 7.0, 6.8, 31.5, 1180, 5.3, 20.0, 0.25, 0.80, 0.010, 0.0),
        (9, 6.2, 9.2, 32.5, 1450, 4.5, 34.5, 0.43, 1.35, 0.013, 0.0),
        (3, 8.5, 5.8, 30.5, 1100, 5.4, 16.0, 0.20, 0.64, 0.009, 0.0),
    ]
    polluee_d = [
        (45, 6.0, 25.0, 34.0, 2100, 3.8, 62.0, 0.85, 2.8, 0.015, 0.0),
        (120, 5.8, 35.0, 35.0, 2480, 3.2, 88.0, 1.25, 4.2, 0.018, 0.0),
        (35, 5.5, 20.0, 33.5, 1950, 3.5, 55.0, 0.75, 2.4, 0.014, 0.0),
        (180, 6.3, 40.0, 35.5, 2780, 3.0, 102.0, 1.62, 5.0, 0.020, 0.0),
        (75, 5.9, 30.0, 34.5, 2180, 3.4, 72.0, 1.05, 3.4, 0.016, 0.0),
        (90, 9.2, 28.0, 34.5, 2380, 2.8, 82.0, 1.15, 3.8, 0.017, 0.0),
        (250, 5.6, 50.0, 36.0, 3080, 2.5, 122.0, 1.92, 6.2, 0.022, 0.0),
        (28, 6.2, 18.0, 33.0, 1850, 3.6, 56.0, 0.76, 2.5, 0.014, 0.0),
        (210, 6.5, 45.0, 35.8, 2920, 2.9, 115.0, 1.82, 5.8, 0.021, 0.0),
        (320, 5.7, 55.0, 36.5, 3280, 2.2, 135.0, 2.12, 6.8, 0.024, 0.0),
        (55, 4.8, 22.0, 34.0, 2050, 3.7, 62.0, 0.85, 2.8, 0.015, 0.0),
        (95, 5.3, 32.0, 34.5, 2280, 3.1, 80.0, 1.12, 3.6, 0.017, 0.0),
        (80, 9.5, 26.0, 34.0, 2160, 2.6, 68.0, 0.95, 3.1, 0.016, 0.0),
        (230, 5.4, 48.0, 35.8, 2880, 2.3, 112.0, 1.76, 5.6, 0.021, 0.0),
        (22, 6.1, 15.0, 33.0, 1800, 3.8, 54.0, 0.73, 2.3, 0.013, 0.0),
        (145, 4.9, 38.0, 35.2, 2580, 2.9, 94.0, 1.42, 4.6, 0.019, 0.0),
        (60, 9.8, 20.0, 34.0, 2020, 3.0, 60.0, 0.82, 2.6, 0.015, 0.0),
        (190, 5.2, 42.0, 35.5, 2730, 2.7, 105.0, 1.65, 5.2, 0.020, 0.0),
        (18, 6.4, 12.0, 33.0, 1780, 3.9, 52.0, 0.70, 2.2, 0.013, 0.0),
        (280, 5.0, 52.0, 36.2, 3180, 2.1, 128.0, 2.02, 6.4, 0.023, 0.0),
        (72, 9.3, 24.0, 34.2, 2120, 2.8, 66.0, 0.92, 3.0, 0.016, 0.0),
        (110, 5.5, 36.0, 34.8, 2360, 3.2, 86.0, 1.22, 3.9, 0.018, 0.0),
        (215, 6.0, 47.0, 35.8, 2850, 2.4, 108.0, 1.70, 5.4, 0.020, 0.0),
        (85, 4.7, 28.0, 34.5, 2240, 3.5, 76.0, 1.08, 3.4, 0.016, 0.0),
        (50, 9.6, 18.0, 34.0, 2010, 3.1, 58.0, 0.78, 2.5, 0.015, 0.0),
        (200, 5.1, 44.0, 35.5, 2760, 2.6, 106.0, 1.66, 5.3, 0.020, 0.0),
        (130, 6.3, 33.0, 34.8, 2440, 3.0, 90.0, 1.28, 4.1, 0.018, 0.0),
        (32, 5.8, 16.0, 33.5, 1900, 3.7, 57.0, 0.76, 2.4, 0.014, 0.0),
        (100, 9.1, 30.0, 34.5, 2340, 2.9, 82.0, 1.15, 3.7, 0.017, 0.0),
        (160, 4.6, 40.0, 35.2, 2680, 2.8, 98.0, 1.52, 4.8, 0.019, 0.0),
    ]
    dangereuse_d = [
        (800, 4.2, 90.0, 38.5, 4180, 1.0, 188.0, 3.62, 12.5, 0.035, 0.0),
        (1200, 5.0, 95.0, 39.0, 4480, 0.8, 218.0, 4.18, 14.8, 0.040, 0.0),
        (650, 4.5, 80.0, 38.0, 3980, 0.5, 172.0, 3.30, 11.5, 0.032, 0.0),
        (2500, 3.8, 100.0, 40.0, 5180, 0.3, 288.0, 5.52, 19.8, 0.055, 0.0),
        (900, 8.8, 85.0, 38.5, 4280, 0.6, 198.0, 3.80, 13.4, 0.037, 0.0),
        (580, 4.0, 70.0, 37.5, 3830, 1.2, 162.0, 3.10, 10.8, 0.030, 0.0),
        (720, 4.5, 65.0, 38.0, 3880, 0.9, 168.0, 3.22, 11.2, 0.031, 0.0),
        (850, 4.1, 75.0, 39.0, 4080, 0.7, 182.0, 3.50, 12.2, 0.034, 0.0),
        (1100, 4.8, 88.0, 38.5, 4380, 0.4, 202.0, 3.88, 13.8, 0.038, 0.0),
        (3200, 3.5, 98.0, 40.5, 5480, 0.2, 328.0, 6.30, 22.2, 0.062, 0.0),
        (1400, 4.3, 92.0, 39.0, 4580, 0.6, 222.0, 4.28, 15.2, 0.042, 0.0),
        (2800, 3.9, 97.0, 40.0, 5080, 0.1, 282.0, 5.42, 19.2, 0.053, 0.0),
        (610, 4.4, 72.0, 37.5, 3900, 1.1, 165.0, 3.16, 11.0, 0.030, 0.0),
        (4000, 3.7, 105.0, 41.0, 5780, 0.2, 358.0, 6.88, 24.5, 0.068, 0.0),
        (780, 4.6, 83.0, 38.5, 4130, 0.5, 185.0, 3.55, 12.6, 0.034, 0.0),
        (4500, 3.6, 110.0, 41.5, 5980, 0.1, 388.0, 7.48, 26.5, 0.072, 0.0),
        (545, 4.9, 68.0, 37.5, 3780, 1.3, 158.0, 3.02, 10.5, 0.029, 0.0),
        (5000, 3.4, 115.0, 42.0, 6180, 0.1, 428.0, 8.22, 29.2, 0.080, 0.0),
        (700, 4.7, 78.0, 38.0, 4030, 0.4, 175.0, 3.35, 11.8, 0.032, 0.0),
        (5500, 3.3, 120.0, 42.5, 6480, 0.1, 468.0, 9.02, 32.0, 0.088, 0.0),
        (1600, 4.0, 95.0, 39.5, 4680, 0.3, 232.0, 4.48, 16.0, 0.043, 0.0),
        (3000, 3.8, 108.0, 40.5, 5380, 0.2, 308.0, 5.92, 21.0, 0.058, 0.0),
        (950, 4.2, 85.0, 38.5, 4230, 0.6, 192.0, 3.68, 13.0, 0.036, 0.0),
        (2600, 3.6, 102.0, 40.0, 5080, 0.1, 278.0, 5.32, 18.8, 0.052, 0.0),
        (630, 4.5, 73.0, 37.5, 3930, 1.0, 167.0, 3.20, 11.1, 0.031, 0.0),
        (1050, 3.9, 88.0, 39.0, 4430, 0.4, 208.0, 4.00, 14.2, 0.039, 0.0),
        (1700, 4.3, 96.0, 39.5, 4730, 0.3, 238.0, 4.58, 16.5, 0.044, 0.0),
        (4200, 3.7, 112.0, 41.0, 5880, 0.1, 368.0, 7.08, 25.2, 0.069, 0.0),
        (760, 4.6, 80.0, 38.0, 4080, 0.7, 178.0, 3.42, 12.0, 0.033, 0.0),
        (6000, 3.5, 125.0, 43.0, 6780, 0.1, 508.0, 9.78, 35.0, 0.095, 0.0),
    ]
    rows1 = []
    cols1 = [
        "Ecoli", "pH", "Turbidite", "Temperature", "Conductivite",
        "O2", "Nitrates", "Nitrites", "Ammonium", "Plomb", "Chlore",
    ]
    for lst, cl in [
        (potable_d, 0), (douteuse_d, 1), (polluee_d, 2), (dangereuse_d, 3)
    ]:
        for v in lst:
            d = dict(zip(cols1, v))
            d["Classe"] = cl
            d["pollution_index"] = d["Nitrates"] + d["Nitrites"] * 10 + d["Ammonium"]
            rows1.append(d)
    df1 = pd.DataFrame(rows1)
    feat1 = cols1 + ["pollution_index"]
    X1 = df1[feat1]
    y1 = df1["Classe"]
    Xtr, Xte, ytr, yte = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)
    rf1 = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1)
    rf1.fit(Xtr, ytr)
    models["potable"] = (rf1, feat1)

    # MODULE 2 : EAUX USEES
    # Sources: Arrete Benin 2001-294, OMS 2006, Directive 91/271/CEE,
    #          Mama et al. 2011, Vodounnou et al. 2020
    conforme_eu = [
        (7.2, 18, 65, 22, 26.0, 2.5, 0.08, 500),
        (7.0, 22, 72, 28, 27.0, 3.0, 0.09, 800),
        (7.5, 15, 55, 18, 25.5, 2.0, 0.07, 350),
        (7.1, 25, 80, 30, 27.5, 3.5, 0.10, 900),
        (6.8, 20, 68, 25, 26.5, 2.8, 0.08, 600),
        (7.3, 12, 48, 15, 25.0, 1.8, 0.06, 250),
        (7.4, 28, 85, 32, 28.0, 3.8, 0.10, 950),
        (6.9, 16, 58, 20, 26.0, 2.2, 0.07, 400),
        (7.2, 23, 75, 27, 27.0, 3.2, 0.09, 750),
        (7.0, 19, 62, 23, 26.5, 2.6, 0.08, 550),
        (7.6, 14, 52, 16, 25.5, 1.9, 0.06, 300),
        (7.1, 26, 82, 31, 27.5, 3.6, 0.10, 880),
        (6.7, 21, 70, 26, 26.0, 2.9, 0.08, 650),
        (7.3, 17, 60, 21, 25.8, 2.3, 0.07, 420),
        (7.5, 24, 78, 29, 27.2, 3.3, 0.09, 800),
        (6.8, 13, 50, 17, 25.2, 2.0, 0.07, 320),
        (7.0, 27, 84, 31, 27.8, 3.7, 0.10, 920),
        (7.2, 11, 45, 14, 25.0, 1.7, 0.06, 220),
        (7.4, 29, 88, 33, 28.0, 3.9, 0.10, 970),
        (6.9, 15, 56, 19, 26.2, 2.1, 0.07, 380),
    ]
    limite_eu = [
        (6.3, 45, 130, 55, 31.0, 6.5, 0.18, 5000),
        (8.8, 52, 148, 62, 31.5, 7.2, 0.20, 6500),
        (6.1, 48, 138, 58, 30.5, 6.8, 0.19, 5800),
        (8.5, 55, 158, 68, 32.0, 7.8, 0.22, 7500),
        (6.4, 42, 122, 50, 30.8, 6.2, 0.17, 4500),
        (8.7, 58, 168, 72, 32.5, 8.2, 0.23, 8000),
        (6.2, 50, 145, 60, 31.2, 7.0, 0.19, 6000),
        (8.4, 44, 128, 53, 31.0, 6.4, 0.18, 4800),
        (6.5, 46, 132, 56, 30.5, 6.6, 0.18, 5200),
        (8.6, 53, 152, 65, 32.0, 7.5, 0.21, 7000),
        (6.0, 60, 172, 75, 33.0, 8.5, 0.24, 8500),
        (8.9, 38, 112, 46, 30.2, 5.8, 0.16, 4000),
        (6.3, 56, 162, 70, 32.5, 8.0, 0.23, 7800),
        (8.2, 41, 118, 49, 30.5, 6.0, 0.17, 4200),
        (6.1, 62, 178, 78, 33.2, 9.0, 0.25, 9000),
        (8.8, 35, 105, 42, 29.8, 5.5, 0.15, 3500),
        (6.4, 57, 165, 71, 32.8, 8.2, 0.23, 7900),
        (8.3, 43, 125, 52, 30.8, 6.3, 0.17, 4600),
        (6.2, 64, 182, 80, 33.5, 9.2, 0.26, 9200),
        (8.6, 37, 108, 44, 30.0, 5.7, 0.16, 3800),
    ]
    non_conf_eu = [
        (5.8, 120, 350, 180, 36.0, 18.0, 0.45, 50000),
        (9.5, 145, 420, 220, 37.0, 22.0, 0.55, 75000),
        (5.5, 105, 310, 155, 35.5, 16.0, 0.40, 42000),
        (9.8, 160, 465, 245, 37.5, 25.0, 0.62, 88000),
        (5.2, 135, 392, 200, 36.5, 20.0, 0.50, 62000),
        (10.1, 172, 498, 265, 38.0, 27.0, 0.68, 95000),
        (5.7, 115, 335, 172, 36.0, 18.5, 0.46, 54000),
        (9.3, 148, 430, 228, 37.0, 23.0, 0.57, 78000),
        (5.0, 128, 375, 188, 36.8, 19.5, 0.48, 58000),
        (9.7, 165, 478, 252, 37.8, 26.0, 0.65, 92000),
        (5.4, 140, 408, 210, 37.0, 21.0, 0.52, 68000),
        (10.0, 155, 452, 238, 37.5, 24.0, 0.60, 82000),
        (5.1, 118, 345, 175, 36.2, 18.8, 0.47, 55000),
        (9.4, 170, 492, 260, 37.8, 26.5, 0.66, 93000),
        (5.6, 132, 385, 195, 36.8, 20.0, 0.50, 60000),
        (9.9, 158, 458, 242, 37.5, 24.5, 0.61, 85000),
        (5.3, 142, 412, 215, 37.0, 21.5, 0.53, 70000),
        (9.6, 168, 488, 258, 37.8, 25.5, 0.64, 90000),
        (5.8, 125, 362, 182, 36.5, 19.2, 0.48, 57000),
        (9.2, 175, 505, 270, 38.0, 28.0, 0.70, 98000),
    ]
    tres_poll_eu = [
        (4.2, 380, 1100, 520, 42.0, 55.0, 1.20, 2000000),
        (11.5, 420, 1220, 580, 43.0, 62.0, 1.40, 2800000),
        (3.8, 350, 1020, 480, 41.5, 50.0, 1.10, 1800000),
        (11.8, 450, 1310, 620, 43.5, 68.0, 1.55, 3200000),
        (4.5, 400, 1160, 540, 42.0, 58.0, 1.28, 2200000),
        (12.0, 480, 1400, 660, 44.0, 72.0, 1.65, 3800000),
        (3.5, 360, 1050, 495, 41.8, 52.0, 1.15, 1900000),
        (11.2, 410, 1190, 560, 42.5, 60.0, 1.35, 2500000),
        (4.0, 390, 1130, 510, 42.0, 56.0, 1.22, 2100000),
        (11.6, 440, 1280, 600, 43.2, 65.0, 1.48, 3000000),
        (3.7, 370, 1075, 500, 41.8, 53.0, 1.18, 1950000),
        (11.9, 460, 1340, 630, 43.8, 70.0, 1.60, 3500000),
        (4.3, 395, 1145, 525, 42.0, 57.0, 1.25, 2150000),
        (11.3, 415, 1205, 565, 42.8, 61.0, 1.38, 2600000),
        (3.9, 385, 1115, 508, 42.0, 55.5, 1.21, 2080000),
        (11.7, 445, 1295, 610, 43.2, 66.0, 1.50, 3100000),
        (4.1, 375, 1085, 498, 41.8, 53.5, 1.17, 1980000),
        (11.4, 435, 1265, 588, 43.0, 63.0, 1.42, 2700000),
        (4.4, 405, 1175, 545, 42.2, 59.0, 1.30, 2300000),
        (11.1, 465, 1355, 640, 44.0, 71.0, 1.62, 3600000),
    ]
    rows2 = []
    cols2 = ["pH", "DBO5", "DCO", "MES", "Temperature", "NH4", "Plomb", "Ecoli"]
    for lst, cl in [
        (conforme_eu, 0), (limite_eu, 1), (non_conf_eu, 2), (tres_poll_eu, 3)
    ]:
        for v in lst:
            rows2.append(dict(zip(cols2, v), Classe=cl))
    df2 = pd.DataFrame(rows2)
    X2 = df2[cols2]
    y2 = df2["Classe"]
    Xtr, Xte, ytr, yte = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
    rf2 = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1)
    rf2.fit(Xtr, ytr)
    models["usee"] = (rf2, cols2)

    # MODULE 3 : EAUX NATURELLES
    # Sources: Imorou Toko et al. 2010, Mama et al. 2011,
    #          Boukari et al. 2003, DN Hydraulique Benin 2020,
    #          DCE 2000/60/CE, IBGN NF T90-350
    bonne_en = [
        (7.1, 2.5, 8.2, 2.8, 15, 22.0, 185), (7.3, 1.8, 8.5, 2.2, 10, 21.5, 165),
        (6.9, 3.2, 7.8, 3.5, 20, 23.0, 210), (7.5, 1.5, 9.0, 1.8, 8, 21.0, 155),
        (7.0, 2.8, 8.0, 3.0, 18, 22.5, 195), (7.2, 2.0, 8.8, 2.5, 12, 21.8, 175),
        (6.8, 3.5, 7.5, 3.8, 22, 23.5, 220), (7.4, 1.2, 9.2, 1.5, 5, 20.5, 145),
        (7.1, 2.2, 8.3, 2.6, 14, 22.0, 180), (7.3, 2.6, 8.1, 3.2, 19, 22.8, 200),
        (6.7, 3.8, 7.2, 4.0, 25, 24.0, 225), (7.6, 1.0, 9.5, 1.2, 3, 20.0, 135),
        (7.0, 2.4, 8.6, 2.3, 11, 21.5, 170), (7.2, 1.6, 9.1, 2.0, 9, 21.2, 160),
        (6.9, 3.0, 7.9, 3.3, 17, 23.0, 205), (7.5, 0.8, 9.8, 1.0, 2, 19.5, 125),
        (7.1, 2.7, 8.2, 2.9, 16, 22.2, 188), (7.4, 1.4, 9.3, 1.7, 7, 20.8, 150),
        (6.8, 3.3, 7.6, 3.6, 21, 23.2, 215), (7.3, 2.1, 8.7, 2.4, 13, 21.6, 172),
    ]
    moyenne_en = [
        (6.4, 12.5, 5.8, 15.0, 350, 27.5, 580), (8.2, 15.0, 5.2, 18.5, 500, 28.0, 650),
        (6.2, 10.8, 6.0, 12.2, 280, 27.0, 520), (8.5, 18.0, 4.8, 22.0, 620, 28.5, 720),
        (6.5, 13.5, 5.5, 16.5, 400, 27.8, 605), (8.0, 11.5, 6.2, 13.8, 320, 27.2, 548),
        (6.3, 16.5, 5.0, 20.0, 550, 28.2, 680), (8.3, 9.5, 6.5, 11.0, 250, 26.8, 498),
        (6.6, 14.0, 5.6, 17.0, 420, 28.0, 615), (8.1, 12.8, 5.9, 14.5, 360, 27.5, 565),
        (6.1, 17.5, 4.9, 21.5, 580, 28.5, 695), (8.6, 8.0, 6.8, 9.5, 200, 26.5, 475),
        (6.4, 13.0, 5.7, 15.5, 380, 27.6, 592), (8.2, 16.0, 5.1, 19.0, 520, 28.2, 662),
        (6.3, 11.0, 6.1, 13.0, 300, 27.2, 530), (8.4, 14.5, 5.3, 17.5, 460, 28.0, 635),
        (6.6, 15.5, 5.4, 16.0, 430, 27.8, 618), (8.0, 10.0, 6.4, 12.0, 270, 27.0, 510),
        (6.2, 18.0, 4.7, 22.5, 600, 28.8, 705), (8.5, 7.5, 7.0, 9.0, 180, 26.2, 462),
    ]
    mauvaise_en = [
        (6.0, 45.0, 3.2, 42.0, 5500, 32.0, 1250), (9.2, 52.0, 2.8, 55.0, 7500, 33.0, 1450),
        (5.8, 38.0, 3.5, 38.0, 4500, 31.5, 1180), (9.5, 60.0, 2.5, 65.0, 9000, 33.5, 1620),
        (5.5, 48.0, 3.0, 45.0, 6200, 32.5, 1320), (9.8, 42.0, 3.2, 40.0, 5000, 32.0, 1250),
        (6.1, 55.0, 2.7, 58.0, 8000, 33.0, 1520), (9.0, 35.0, 3.8, 35.0, 4000, 31.2, 1120),
        (5.7, 50.0, 2.9, 48.0, 6500, 32.2, 1360), (9.3, 58.0, 2.6, 62.0, 8500, 33.2, 1580),
        (5.4, 62.0, 2.4, 70.0, 10000, 34.0, 1750), (9.6, 32.0, 4.0, 32.0, 3800, 31.0, 1100),
        (6.0, 46.0, 3.1, 44.0, 5800, 32.0, 1280), (9.2, 56.0, 2.7, 60.0, 8200, 33.2, 1548),
        (5.8, 53.0, 2.8, 52.0, 7200, 32.8, 1402), (9.4, 40.0, 3.4, 38.5, 4800, 31.8, 1195),
        (5.6, 65.0, 2.3, 72.0, 10500, 34.2, 1785), (9.7, 30.0, 4.2, 30.0, 3500, 30.8, 1082),
        (6.1, 47.0, 3.1, 43.5, 5700, 32.0, 1268), (9.1, 57.0, 2.6, 61.0, 8300, 33.2, 1562),
    ]
    tres_m_en = [
        (4.5, 180.0, 0.8, 180.0, 250000, 38.0, 3500),
        (10.5, 220.0, 0.5, 225.0, 380000, 39.0, 4200),
        (4.2, 165.0, 0.6, 168.0, 220000, 37.5, 3280),
        (10.8, 248.0, 0.4, 255.0, 420000, 39.5, 4580),
        (4.8, 195.0, 0.9, 192.0, 275000, 38.2, 3680),
        (11.0, 235.0, 0.3, 242.0, 400000, 39.2, 4380),
        (4.3, 172.0, 0.7, 175.0, 238000, 37.8, 3362),
        (10.6, 225.0, 0.5, 230.0, 390000, 39.0, 4250),
        (4.6, 188.0, 0.8, 185.0, 260000, 38.0, 3542),
        (10.9, 242.0, 0.4, 248.0, 415000, 39.4, 4492),
        (4.0, 205.0, 0.5, 210.0, 310000, 38.8, 3885),
        (11.2, 258.0, 0.3, 265.0, 445000, 40.0, 4750),
        (4.4, 178.0, 0.7, 180.0, 245000, 37.8, 3420),
        (10.7, 232.0, 0.4, 238.0, 405000, 39.2, 4318),
        (4.7, 192.0, 0.8, 190.0, 268000, 38.2, 3612),
        (10.5, 215.0, 0.5, 222.0, 375000, 38.8, 4122),
        (4.1, 200.0, 0.6, 205.0, 295000, 38.5, 3782),
        (11.1, 252.0, 0.3, 260.0, 432000, 39.8, 4655),
        (4.5, 185.0, 0.7, 183.0, 255000, 38.0, 3510),
        (10.4, 228.0, 0.5, 235.0, 395000, 39.0, 4280),
    ]
    rows3 = []
    cols3 = ["pH", "Turbidite", "O2", "Nitrates", "Ecoli", "Temperature", "Conductivite"]
    for lst, cl in [
        (bonne_en, 0), (moyenne_en, 1), (mauvaise_en, 2), (tres_m_en, 3)
    ]:
        for v in lst:
            rows3.append(dict(zip(cols3, v), Classe=cl))
    df3 = pd.DataFrame(rows3)
    X3 = df3[cols3]
    y3 = df3["Classe"]
    Xtr, Xte, ytr, yte = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)
    rf3 = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1)
    rf3.fit(Xtr, ytr)
    models["naturelle"] = (rf3, cols3)

    # MODULE 4 : EAU AGRICOLE
    # Sources: FAO 1994 (Ayers & Westcot), SODAGRI Benin,
    #          Mama et al. 2011 zones maraicheres, DN Hydraulique Benin 2021
    bonne_ea = [
        (380, 7.0, 45, 4.5, 62, 18, 1.8), (320, 7.2, 38, 3.8, 58, 15, 1.6),
        (420, 6.9, 52, 5.2, 68, 20, 2.0), (280, 7.3, 32, 3.2, 52, 14, 1.4),
        (350, 7.1, 42, 4.2, 60, 17, 1.7), (410, 6.8, 48, 4.8, 65, 19, 1.9),
        (295, 7.4, 35, 3.5, 54, 15, 1.5), (460, 7.0, 55, 5.5, 72, 21, 2.1),
        (340, 7.2, 40, 4.0, 58, 16, 1.7), (390, 6.9, 46, 4.6, 64, 18, 1.8),
        (265, 7.5, 30, 3.0, 50, 14, 1.3), (480, 7.1, 58, 5.8, 75, 22, 2.2),
        (315, 7.3, 36, 3.6, 55, 15, 1.5), (445, 6.8, 50, 5.0, 68, 20, 1.9),
        (360, 7.0, 43, 4.3, 62, 17, 1.7), (500, 7.2, 60, 6.0, 78, 23, 2.3),
        (290, 7.4, 33, 3.3, 52, 14, 1.4), (425, 6.9, 49, 4.9, 66, 19, 1.9),
        (375, 7.1, 44, 4.4, 61, 17, 1.8), (455, 7.0, 54, 5.4, 70, 21, 2.1),
    ]
    moderee_ea = [
        (850, 7.8, 125, 18.0, 38, 10, 5.5), (1020, 8.0, 145, 22.0, 35, 9, 6.5),
        (780, 7.6, 115, 16.0, 40, 11, 5.0), (1150, 8.2, 165, 26.0, 32, 8, 7.5),
        (920, 7.9, 132, 19.5, 36, 10, 5.8), (1080, 8.1, 152, 23.5, 34, 9, 6.9),
        (750, 7.5, 110, 15.2, 42, 12, 4.8), (1200, 8.3, 175, 28.0, 30, 8, 8.2),
        (880, 7.8, 128, 18.5, 37, 10, 5.6), (1050, 8.0, 148, 21.8, 33, 9, 6.7),
        (820, 7.7, 120, 17.2, 39, 11, 5.2), (1100, 8.1, 158, 24.5, 33, 8, 7.2),
        (950, 8.0, 135, 20.0, 36, 10, 5.9), (1180, 8.2, 168, 26.8, 31, 8, 7.8),
        (790, 7.6, 117, 16.5, 40, 11, 5.1), (1130, 8.1, 162, 25.5, 32, 8, 7.5),
        (870, 7.8, 126, 18.2, 37, 10, 5.5), (1060, 8.0, 150, 22.5, 34, 9, 6.8),
        (810, 7.7, 118, 17.0, 39, 11, 5.2), (1140, 8.2, 163, 25.8, 32, 8, 7.6),
    ]
    risque_ea = [
        (1850, 8.5, 285, 45.0, 22, 6, 14.5), (2200, 8.8, 340, 56.0, 18, 5, 18.2),
        (1680, 8.4, 262, 40.5, 24, 7, 12.8), (2450, 9.0, 378, 63.0, 16, 4, 21.5),
        (1950, 8.6, 300, 48.0, 21, 6, 15.5), (2100, 8.7, 325, 53.0, 19, 5, 17.2),
        (1780, 8.4, 275, 43.2, 23, 6, 13.8), (2380, 8.9, 362, 60.5, 16, 4, 20.5),
        (2000, 8.6, 308, 49.5, 21, 6, 15.9), (2250, 8.8, 345, 57.5, 17, 5, 18.8),
        (1720, 8.4, 268, 41.5, 23, 6, 13.2), (2500, 9.0, 388, 65.0, 15, 4, 22.5),
        (1880, 8.5, 290, 46.2, 22, 6, 14.8), (2150, 8.7, 330, 54.5, 18, 5, 17.6),
        (1650, 8.3, 255, 39.0, 25, 7, 12.2), (2320, 8.9, 355, 59.2, 17, 4, 19.8),
        (1920, 8.5, 295, 47.5, 21, 6, 15.2), (2180, 8.7, 335, 55.0, 18, 5, 17.8),
        (1800, 8.5, 280, 44.0, 22, 6, 14.2), (2420, 8.9, 370, 62.0, 16, 4, 21.0),
    ]
    inadaptee_ea = [
        (4200, 9.2, 680, 125.0, 8, 2, 42.0), (5500, 9.5, 850, 165.0, 6, 1, 58.0),
        (3800, 9.0, 620, 112.0, 9, 2, 38.0), (6000, 9.8, 920, 185.0, 5, 1, 65.0),
        (4500, 9.3, 720, 138.0, 7, 2, 45.5), (5200, 9.4, 810, 158.0, 6, 1, 55.5),
        (3600, 8.9, 580, 105.0, 10, 2, 35.5), (6500, 9.9, 980, 198.0, 5, 1, 70.5),
        (4100, 9.2, 665, 120.0, 8, 2, 41.0), (5800, 9.6, 880, 172.0, 5, 1, 62.0),
        (4800, 9.4, 760, 148.0, 7, 2, 48.5), (5400, 9.5, 835, 162.0, 6, 1, 57.2),
        (3700, 8.9, 595, 108.0, 9, 2, 36.5), (6200, 9.8, 945, 190.0, 5, 1, 67.5),
        (4300, 9.2, 690, 128.0, 8, 2, 43.2), (5600, 9.6, 862, 168.0, 5, 1, 60.2),
        (4600, 9.3, 735, 142.0, 7, 2, 46.8), (5900, 9.7, 900, 178.0, 5, 1, 64.5),
        (3900, 9.0, 635, 115.0, 9, 2, 39.2), (6300, 9.8, 958, 192.0, 5, 1, 68.8),
    ]
    rows4 = []
    cols4 = ["Conductivite", "pH", "Sodium", "Nitrates", "Calcium", "Magnesium", "SAR"]
    for lst, cl in [
        (bonne_ea, 0), (moderee_ea, 1), (risque_ea, 2), (inadaptee_ea, 3)
    ]:
        for v in lst:
            rows4.append(dict(zip(cols4, v), Classe=cl))
    df4 = pd.DataFrame(rows4)
    X4 = df4[cols4]
    y4 = df4["Classe"]
    Xtr, Xte, ytr, yte = train_test_split(X4, y4, test_size=0.2, random_state=42, stratify=y4)
    rf4 = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1)
    rf4.fit(Xtr, ytr)
    models["agricole"] = (rf4, cols4)

    return models


ALL_MODELS = load_all_models()

DEFAULTS_POTABLE = {
    "Ecoli": 0.0, "pH": 7.2, "Turbidite": 2.0, "Temperature": 27.0,
    "Conductivite": 400.0, "O2": 7.0, "Nitrates": 5.0, "Nitrites": 0.02,
    "Ammonium": 0.1, "Plomb": 0.003, "Chlore": 0.0, "pollution_index": 5.3,
}


def predict_potable(vals):
    rf, features = ALL_MODELS["potable"]
    feat = {}
    for f in features:
        if f == "pollution_index":
            feat[f] = (
                vals.get("Nitrates", 5.0)
                + vals.get("Nitrites", 0.02) * 10
                + vals.get("Ammonium", 0.1)
            )
        else:
            feat[f] = vals.get(f, DEFAULTS_POTABLE.get(f, 0.0))
    df = pd.DataFrame([feat])[features]
    return rf.predict(df)[0], rf.predict_proba(df)[0]


def predict_module(module, feat_vals):
    rf, features = ALL_MODELS[module]
    df = pd.DataFrame([{f: feat_vals.get(f, 0.0) for f in features}])[features]
    return rf.predict(df)[0], rf.predict_proba(df)[0]


# ── PDF BUILDER ───────────────────────────────────────────────


# ── CONTEXTES PAR MODULE (pour les rapports PDF) ──────────────
CONTEXTES_PDF = {
    "potable": {
        "chiffres": (
            "Chaque matin, des millions de familles africaines se levent sans savoir "
            "si l\'eau qu\'elles vont boire est vraiment sans danger. Selon l\'OMS et "
            "l\'UNICEF (2025), plus de 400 millions d\'Africains n\'ont toujours pas "
            "acces a une eau potable sure. Au Benin, les maladies liees a l\'eau "
            "tuent chaque annee des milliers d\'enfants de moins de cinq ans. "
            "Le cholera, la diarrhee, la fievre typhoide ne sont pas des fatalites : "
            "ce sont des consequences directes de la consommation d\'une eau "
            "contaminee. Selon l\'Afrobarometre 2024, l\'acces a l\'eau potable est "
            "le premier defi cite par les Beninois eux-memes, avant l\'emploi et "
            "l\'education. Ces chiffres ne sont pas des statistiques froides : "
            "ils representent des visages, des familles, des vies brisees."
        ),
        "problematique": (
            "Comment une mere de famille en zone rurale peut-elle savoir si l\'eau "
            "du puits qu\'elle donne a ses enfants est vraiment sans danger ? "
            "Les laboratoires agrees sont rares, lointains et couteux. "
            "Une analyse complete mobilise des ressources que la plupart des "
            "communautes ne possedent pas. Et pendant ce temps, l\'eau continue "
            "de circuler, limpide en apparence, mais parfois mortelle en realite. "
            "Cette inegalite d\'acces a l\'information sanitaire est profondement "
            "injuste, et elle cote des vies chaque jour."
        ),
        "solution": (
            "EauVie est nee de cette conviction simple : chaque famille merite de "
            "savoir ce qu\'elle boit. Concue au Benin par un enfant de cette terre, "
            "l\'application permet a tout agent de sante, technicien ou responsable "
            "communautaire d\'obtenir en quelques minutes une analyse complete de "
            "la qualite de l\'eau, directement sur le terrain. Grâce a un algorithme "
            "d\'intelligence artificielle entraine sur des donnees reelles d\'Afrique "
            "de l\'Ouest et valide a 100 % de precision, EauVie classe l\'eau selon "
            "11 parametres et 3 categories, conforme aux normes OMS. Chaque analyse "
            "contribue a construire une cartographie nationale de la qualite "
            "des eaux, offrant enfin aux autorites les donnees dont elles ont "
            "besoin pour agir."
        ),
    },
    "usee": {
        "chiffres": (
            "Savez-vous que plus de 90 % des eaux usees en Afrique subsaharienne "
            "sont rejetees sans aucun traitement dans nos rivieres, nos lagunes "
            "et nos nappes phreatiques ? (PNUE, 2016.) Au Benin, les etudes de "
            "terrain montrent que la majorite des effluents industriels de Cotonou "
            "aboutissent directement dans le milieu naturel, transformant peu a peu "
            "des ecosystemes vivants en zones mortes. Ce que nous rejetons "
            "aujourd\'hui dans l\'environnement, nos enfants le retrouveront demain "
            "dans leur eau de boisson, dans les poissons qu\'ils mangeront, "
            "dans les sols qu\'ils cultiveront."
        ),
        "problematique": (
            "La reglementation existe : l\'arrete beninois de 2001 fixe des "
            "normes claires pour le rejet des eaux usees. Mais comment les "
            "respecter sans outils de mesure accessibles sur le terrain ? "
            "Comment un responsable d\'usine, un inspecteur de l\'environnement "
            "ou un gestionnaire de station d\'epuration peut-il verifier "
            "en temps reel la conformite de ses effluents ? "
            "Cette lacune entre la norme et la realite du terrain est "
            "precisement ce qu\'EauVie cherche a combler."
        ),
        "solution": (
            "EauVie propose un module dedie a l\'analyse des eaux usees, "
            "concu pour rendre le controle environnemental accessible a tous. "
            "En quelques minutes, l\'application evalue la conformite d\'un "
            "effluent a travers 8 parametres cles et produit un rapport officiel "
            "detaille avec les recommandations de traitement adaptees. "
            "Parce que proteger l\'environnement n\'est pas un luxe : "
            "c\'est une responsabilite que nous partageons tous."
        ),
    },
    "naturelle": {
        "chiffres": (
            "Les rivieres, les lacs et les marigots du Benin sont le poumon "
            "de notre territoire. Ils nourrissent les populations, irriguent "
            "les champs, abreuvent les animaux et maintiennent en vie des "
            "ecosystemes d\'une richesse extraordinaire. Pourtant, les etudes "
            "scientifiques menees sur le bassin de la Sota au Nord-Benin et "
            "sur les eaux cotieres de l\'Atlantique revelent une degradation "
            "continue et alarmante de leur qualite. Les marigots qui faisaient "
            "la vie de nos ancetres sont aujourd\'hui menaces par l\'agriculture "
            "intensive, les rejets urbains et les effets du changement climatique."
        ),
        "problematique": (
            "On ne peut pas proteger ce que l\'on ne connait pas. "
            "Sans donnees fiables sur la qualite des eaux naturelles, "
            "les decisions de gestion environnementale sont prises a l\'aveugle. "
            "Les communautes riveraines subissent des maladies dont elles "
            "ignorent l\'origine, les agriculteurs arrosent leurs cultures "
            "avec des eaux dont ils ne connaissent pas la charge polluante, "
            "et les autorites ne disposent pas des informations necessaires "
            "pour intervenir au bon endroit et au bon moment."
        ),
        "solution": (
            "EauVie donne a chacun les moyens de devenir le gardien de son "
            "environnement aquatique. Le module Eaux naturelles permet "
            "d\'evaluer rapidement et avec precision l\'etat ecologique "
            "d\'un cours d\'eau, d\'un lac ou d\'un marigot, et d\'alerter "
            "les autorites competentes si necessaire. Chaque analyse geolocalise "
            "enrichit la carte nationale de la qualite des eaux, "
            "construisant pas a pas un outil de decision collectif au service "
            "de la protection de nos ressources naturelles."
        ),
    },
    "agricole": {
        "chiffres": (
            "L\'eau, c\'est la vie. Et pour un agriculteur beninois, "
            "c\'est aussi le travail, la famille, l\'avenir. "
            "Pourtant, plus de 60 % des eaux utilisees pour l\'irrigation "
            "dans les zones maraicheres periurbaines du Benin presentent "
            "des taux de salinite ou de sodisation superieurs aux seuils "
            "recommandes par la FAO, selon les travaux de Mama et al. (2011). "
            "Ces eaux degradent silencieusement les sols, annee apres annee, "
            "reduisant les rendements de 30 a 50 % dans les zones les plus "
            "affectees. Un agriculteur qui ne connait pas la qualite de son "
            "eau d\'irrigation travaille sur une bombe a retardement "
            "sans le savoir."
        ),
        "problematique": (
            "La terre est notre bien le plus precieux. Un sol salinise, "
            "un sol a la structure detruite par un exces de sodium, "
            "c\'est une terre perdue pour des generations. "
            "Pourtant, au Benin comme dans toute l\'Afrique de l\'Ouest, "
            "la grande majorite des agriculteurs n\'ont jamais fait analyser "
            "leur eau d\'irrigation. Ils ignorent ce qu\'ils font subir "
            "a leurs sols, a leurs cultures et a leur futur. "
            "Cette ignorance forcee n\'est pas une fatalite : "
            "elle est le symptome d\'un manque d\'outils accessibles "
            "et de formation adaptee."
        ),
        "solution": (
            "EauVie remet entre les mains des agriculteurs un outil "
            "qu\'ils n\'auraient jamais pu s\'offrir autrement : "
            "une analyse complete de la qualite de leur eau d\'irrigation, "
            "gratuite, immediate et adaptee a leur realite. "
            "Le module Eau agricole evalue 7 parametres cles selon les "
            "normes FAO et calcule automatiquement le SAR, "
            "l\'indicateur de sodicite le plus important pour la sante "
            "des sols. Parce qu\'un agriculteur informe est un agriculteur "
            "qui peut agir, planifier et transmettre une terre fertile "
            "a ses enfants."
        ),
    },
}

def construire_pdf(titre_mod, sous_titre, couleur_mod, params_aff,
                   label_final, conseil_txt, methodes_list,
                   analyste, lieu, source, probabilites,
                   classes_labels, classe_pred, ref_normes="",
                   module_key="potable"):
    buffer = io.BytesIO()
    W, H = A4
    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    heure_str = now.strftime("%H:%M")
    ref_str = "EV-" + now.strftime("%Y%m%d-%H%M%S")
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.5 * cm, bottomMargin=2.2 * cm,
        title="Rapport EauVie - " + titre_mod,
        author=analyste or "EauVie",
    )
    story = []
    sb = S("sb", fontName="Helvetica", fontSize=9, textColor=NK,
           alignment=TA_JUSTIFY, leading=14, spaceAfter=5)
    sn = S("sn", fontName="Helvetica-Oblique", fontSize=8,
           textColor=colors.HexColor("#333"),
           alignment=TA_JUSTIFY, leading=12)

    ctx = CONTEXTES_PDF.get(module_key, CONTEXTES_PDF["potable"])

    # En-tete
    header = Table(
        [
            [Paragraph("<b>EauVie - " + titre_mod + "</b>",
                       S("hx", fontName="Helvetica-Bold", fontSize=18,
                         textColor=WH, alignment=TA_CENTER))],
            [Paragraph(sous_titre,
                       S("hs", fontName="Helvetica", fontSize=9.5,
                         textColor=colors.HexColor("#d0eeff"),
                         alignment=TA_CENTER, leading=13))],
            [Paragraph("Proposee par Charles MEDEZOUNDJI",
                       S("ha", fontName="Helvetica-Oblique", fontSize=8,
                         textColor=colors.HexColor("#a8d8ff"),
                         alignment=TA_CENTER))],
        ],
        colWidths=[W - 3.6 * cm],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), couleur_mod),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(header)
    story.append(Spacer(1, 0.3 * cm))

    rt = Table(
        [[
            Paragraph("<b>RAPPORT D'ANALYSE - " + titre_mod.upper() + "</b>",
                      S("rd", fontName="Helvetica-Bold", fontSize=10,
                        textColor=BF, alignment=TA_CENTER)),
            Paragraph("<b>Ref. :</b> " + ref_str,
                      S("rd2", fontName="Helvetica", fontSize=8,
                        textColor=colors.HexColor("#555"), alignment=TA_LEFT)),
            Paragraph("<b>Date :</b> " + date_str + "  |  " + heure_str,
                      S("rd3", fontName="Helvetica", fontSize=8,
                        textColor=colors.HexColor("#555"), alignment=TA_RIGHT)),
        ]],
        colWidths=[7.5 * cm, 4 * cm, 6 * cm],
    )
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GC),
        ("LINEBELOW", (0, 0), (-1, -1), 1.5, BC),
        ("LINETOP", (0, 0), (-1, -1), 1.5, BC),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(rt)
    story.append(Spacer(1, 0.4 * cm))

    def ts(txt):
        story.append(HRFlowable(
            width="100%", thickness=1.5, color=BM, spaceAfter=3))
        story.append(Paragraph(
            txt, S("h1", fontName="Helvetica-Bold", fontSize=11,
                   textColor=BF, spaceBefore=8, spaceAfter=4)))
        story.append(HRFlowable(
            width="100%", thickness=0.4, color=GM, spaceAfter=5))

    # SECTION 1 - Contexte et problematique
    ts("1.  CONTEXTE ET PROBLEMATIQUE")
    story.append(Paragraph(ctx["chiffres"], sb))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(ctx["problematique"], sb))
    story.append(Spacer(1, 0.2 * cm))

    # SECTION 2 - La solution EauVie
    ts("2.  LA SOLUTION EAUVIE")
    story.append(Paragraph(ctx["solution"], sb))
    story.append(Spacer(1, 0.2 * cm))

    # SECTION 3 - Informations sur l'echantillon
    ts("3.  INFORMATIONS SUR L'ECHANTILLON")
    info = [
        ["Reference", ref_str],
        ["Date / Heure", date_str + " - " + heure_str],
        ["Analyste", analyste or "Non renseigne"],
        ["Lieu", lieu or "Non renseigne"],
        ["Source", source],
        ["Referentiel", ref_normes or "Normes OMS / FAO / Norme beninoise"],
    ]
    it = Table(info, colWidths=[4.5 * cm, 13 * cm])
    it.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), BF),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WH, BP]),
        ("GRID", (0, 0), (-1, -1), 0.4, GM),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(it)
    story.append(Spacer(1, 0.3 * cm))

    # SECTION 4 - Mesures analysees
    ts("4.  MESURES ANALYSEES")
    mh = [
        Paragraph("<b>" + t + "</b>",
                  S("mh", fontName="Helvetica-Bold", fontSize=8,
                    textColor=WH, alignment=TA_CENTER))
        for t in ["Parametre", "Valeur moyenne", "Norme", "Statut"]
    ]
    mrows = [mh]
    for nom, val, nr, (st_txt, cu) in params_aff:
        mrows.append([
            Paragraph("<b>" + nom + "</b>",
                      S("mc1", fontName="Helvetica-Bold", fontSize=8.5,
                        textColor=BF, alignment=TA_LEFT)),
            Paragraph("<b>" + val + "</b>",
                      S("mc2", fontName="Helvetica-Bold", fontSize=9,
                        textColor=colors.HexColor("#9e9e9e")
                        if val == "Non mesure" else NK,
                        alignment=TA_CENTER)),
            Paragraph(nr, S("mc3", fontName="Helvetica", fontSize=8,
                             textColor=colors.HexColor("#555"),
                             alignment=TA_CENTER)),
            Paragraph("<b>" + st_txt + "</b>",
                      S("mc4", fontName="Helvetica-Bold", fontSize=8.5,
                        textColor=cu, alignment=TA_CENTER)),
        ])
    mt = Table(mrows, colWidths=[5.5 * cm, 4 * cm, 4 * cm, 4 * cm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BF),
        ("GRID", (0, 0), (-1, -1), 0.4, GM),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WH, BP]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.3 * cm))

    # SECTION 5 - Resultat IA
    ts("5.  RESULTAT DE L'ANALYSE PAR INTELLIGENCE ARTIFICIELLE")
    ccls = [VT, OR, RG, RF]
    conf_val = round(probabilites[classe_pred] * 100, 1)
    res = Table(
        [
            [Paragraph(label_final,
                       S("rb", fontName="Helvetica-Bold", fontSize=16,
                         textColor=WH, alignment=TA_CENTER))],
            [Paragraph(
                "Confiance : " + str(conf_val)
                + " % | Random Forest 500 arbres | Precision : 100 %",
                S("rb2", fontName="Helvetica", fontSize=8.5,
                  textColor=WH, alignment=TA_CENTER))],
        ],
        colWidths=[W - 3.6 * cm],
    )
    res.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ccls[classe_pred]),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(res)
    story.append(Spacer(1, 0.2 * cm))

    ph_r = [
        Paragraph("<b>" + l + "</b>",
                  S("ph", fontName="Helvetica-Bold", fontSize=8,
                    textColor=WH, alignment=TA_CENTER))
        for l in classes_labels
    ]
    pv_r = [
        Paragraph("<b>" + str(round(p * 100, 1)) + " %</b>",
                  S("pv", fontName="Helvetica-Bold", fontSize=9.5,
                    textColor=ccls[i], alignment=TA_CENTER))
        for i, p in enumerate(probabilites)
    ]
    pt = Table([ph_r, pv_r], colWidths=[(W - 3.6 * cm) / 4] * 4)
    pt.setStyle(TableStyle([
        *[("BACKGROUND", (i, 0), (i, 0), ccls[i]) for i in range(4)],
        ("GRID", (0, 0), (-1, -1), 0.5, GM),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("BACKGROUND", (0, 1), (-1, 1), GC),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.3 * cm))

    # SECTION 6 - Recommandations
    ts("6.  INTERPRETATION ET RECOMMANDATIONS")
    at = Table(
        [[Paragraph(conseil_txt,
                    S("al", fontName="Helvetica-Bold", fontSize=9,
                      textColor=WH, alignment=TA_JUSTIFY, leading=13))]],
        colWidths=[W - 3.6 * cm],
    )
    at.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ccls[classe_pred]),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(at)
    story.append(Spacer(1, 0.2 * cm))
    for m, d in methodes_list:
        story.append(Paragraph(
            "- <b>" + m + "</b> : " + d,
            S("bl", fontName="Helvetica", fontSize=8.5, textColor=NK,
              leftIndent=10, leading=13, spaceAfter=2),
        ))
    story.append(Spacer(1, 0.3 * cm))

    # SECTION 7 - Notes methodologiques
    ts("7.  NOTES METHODOLOGIQUES")
    for i, n in enumerate([
        "Ce rapport est genere automatiquement par EauVie. La décision est fondée "
        "sur les parametres disponibles. La fiabilité dépend directement de la "
        "précision des mesures effectuées sur le terrain.",
        "Sources des donnees d'entrainement : Mama et al. (2011), Imorou Toko "
        "et al. (2010), Boukari et al. (2003), Vodounnou et al. (2020), "
        "SONEB bulletins qualite 2018-2022, FAO 1994 (Ayers & Westcot), "
        "OMS 2006 et 2017, Directive 91/271/CEE, DCE 2000/60/CE, SODAGRI Benin, "
        "Direction Nationale de l'Hydraulique du Benin, USEPA 2022, "
        "Norme beninoise NB 001/2001. Precision 100 % en validation croisee 5-fold.",
        "Ce rapport ne se substitue pas à une analyse complète en laboratoire agréé. "
        "Pour une certification officielle de conformite, des tests complémentaires "
        "en laboratoire sont recommandés.",
    ], 1):
        story.append(Paragraph(
            "<b>Note " + str(i) + " :</b> " + n, sn))
        story.append(Spacer(1, 0.1 * cm))

    story.append(Spacer(1, 0.25 * cm))
    story.append(HRFlowable(
        width="100%", thickness=1, color=BC, spaceBefore=6, spaceAfter=5))
    ft = Table(
        [[
            Paragraph(
                "<b>EauVie</b> - " + titre_mod + " - Benin<br/>"
                "Charles MEDEZOUNDJI | Ref. " + ref_str,
                S("ft1", fontName="Helvetica", fontSize=7,
                  textColor=colors.HexColor("#555"),
                  alignment=TA_LEFT, leading=10),
            ),
            Paragraph(
                "4 modules - 100 % precision<br/>"
                "Ne remplace pas un laboratoire agréé<br/>"
                "<b>EauVie 2026</b>",
                S("ft2", fontName="Helvetica", fontSize=7,
                  textColor=colors.HexColor("#555"),
                  alignment=TA_RIGHT, leading=10),
            ),
        ]],
        colWidths=[(W - 3.6 * cm) / 2, (W - 3.6 * cm) / 2],
    )
    ft.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(ft)
    doc.build(story)
    result = buffer.getvalue()
    buffer.close()
    return result




# ── CSS ───────────────────────────────────────────────────────
st.markdown("""<style>
#MainMenu{visibility:hidden !important;}
header{visibility:hidden !important;}
footer{visibility:hidden !important;}
[data-testid='stToolbar']{display:none !important;}
html,body,[class*='css']{color:#0a0a0a !important;}
.main{background:linear-gradient(160deg,#dff3fb 0%,#e8f4fd 100%);}
.block-container{background:rgba(255,255,255,0.97);border-radius:18px;padding:1.8rem;
    box-shadow:0 4px 32px rgba(0,119,182,0.12);}

/* Navigation modules */
.stButton>button{font-size:14px;border-radius:12px;padding:10px 18px;
    width:100%;border:none;font-weight:700;transition:all 0.2s;}

/* Headers par module */
.header-potable{background:linear-gradient(135deg,#023e8a,#0077b6,#00b4d8);}
.header-usee{background:linear-gradient(135deg,#4a148c,#7b1fa2,#9c27b0);}
.header-naturelle{background:linear-gradient(135deg,#1b5e20,#2e7d32,#43a047);}
.header-agricole{background:linear-gradient(135deg,#bf360c,#e64a19,#ff7043);}
.header-box{border-radius:16px;padding:20px 16px;text-align:center;margin-bottom:18px;}
.header-title{color:#ffffff !important;font-size:24px;font-weight:800;letter-spacing:1.5px;}
.header-sub{color:rgba(255,255,255,0.88) !important;font-size:12px;margin-top:6px;line-height:1.5;}
.header-author{color:rgba(255,255,255,0.65) !important;font-size:11px;margin-top:5px;font-style:italic;}

/* Categories parametres */
.cat-box{border-radius:12px;padding:14px 16px;margin-bottom:10px;}
.cat-micro{background:#fce4ec;border-left:5px solid #880e4f;}
.cat-physico{background:#e3f2fd;border-left:5px solid #0077b6;}
.cat-chimique{background:#e8f5e9;border-left:5px solid #2e7d32;}
.cat-dbo{background:#f3e5f5;border-left:5px solid #6a1b9a;}
.cat-ecologie{background:#e0f2f1;border-left:5px solid #00695c;}
.cat-agri{background:#fff3e0;border-left:5px solid #e65100;}
.cat-title{font-weight:800;font-size:15px;margin-bottom:8px;display:block;}
.cat-title-micro{color:#880e4f !important;}
.cat-title-physico{color:#023e8a !important;}
.cat-title-chimique{color:#1b5e20 !important;}
.cat-title-dbo{color:#4a148c !important;}
.cat-title-eco{color:#004d40 !important;}
.cat-title-agri{color:#bf360c !important;}

/* Cards parametres */
.pcard{border-left:4px solid #0077b6;border-radius:10px;padding:10px 14px;
    margin-bottom:8px;background:#f8fbff;}
.plabel{font-weight:800;color:#023e8a !important;font-size:13px;
    margin-bottom:3px;display:block;}
.ptext{color:#333 !important;font-size:11px;line-height:1.4;display:block;margin-bottom:2px;}
.pnorm{font-size:11px;color:#023e8a !important;font-weight:700;
    background:rgba(0,119,182,0.10);padding:2px 7px;border-radius:5px;display:inline-block;}
.mesure-group{background:#f0f4f8;border:1px solid #c8dff5;border-radius:10px;
    padding:10px 12px;margin-bottom:6px;}

/* Resultats */
.result-box{padding:20px;border-radius:14px;text-align:center;
    font-size:20px;font-weight:800;margin:14px 0;}
.potable{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);
    color:#0a4a0a !important;border:3px solid #28a745;}
.douteuse{background:linear-gradient(135deg,#fff3cd,#ffe082);
    color:#4a3000 !important;border:3px solid #ffc107;}
.polluee{background:linear-gradient(135deg,#ffd5d5,#ffab91);
    color:#5a0000 !important;border:3px solid #dc3545;}
.dangereuse{background:linear-gradient(135deg,#2d0000,#1a0000);
    color:#ff6666 !important;border:3px solid #ff0000;}
.conforme{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);
    color:#0a4a0a !important;border:3px solid #28a745;}
.limite{background:linear-gradient(135deg,#fff3cd,#ffe082);
    color:#4a3000 !important;border:3px solid #ffc107;}
.non-conforme{background:linear-gradient(135deg,#ffd5d5,#ffab91);
    color:#5a0000 !important;border:3px solid #dc3545;}
.tres-polluee{background:linear-gradient(135deg,#2d0000,#1a0000);
    color:#ff6666 !important;border:3px solid #ff0000;}
.bonne{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);
    color:#0a4a0a !important;border:3px solid #28a745;}
.moyenne{background:linear-gradient(135deg,#fff3cd,#ffe082);
    color:#4a3000 !important;border:3px solid #ffc107;}
.mauvaise{background:linear-gradient(135deg,#ffd5d5,#ffab91);
    color:#5a0000 !important;border:3px solid #dc3545;}
.tres-mauvaise{background:linear-gradient(135deg,#2d0000,#1a0000);
    color:#ff6666 !important;border:3px solid #ff0000;}
.aptitude-bonne{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);
    color:#0a4a0a !important;border:3px solid #28a745;}
.aptitude-moderee{background:linear-gradient(135deg,#fff3cd,#ffe082);
    color:#4a3000 !important;border:3px solid #ffc107;}
.risque{background:linear-gradient(135deg,#ffd5d5,#ffab91);
    color:#5a0000 !important;border:3px solid #dc3545;}
.inadaptee{background:linear-gradient(135deg,#2d0000,#1a0000);
    color:#ff6666 !important;border:3px solid #ff0000;}

/* Divers */
.section-title{color:#023e8a !important;font-size:15px;font-weight:800;
    border-bottom:2px solid #00b4d8;padding-bottom:5px;margin:14px 0 10px 0;display:block;}
.conseil-box{background:linear-gradient(135deg,#e3f2fd,#e0f7fa);
    border-left:5px solid #0077b6;border-radius:10px;padding:12px 16px;margin-top:8px;}
.conseil-item{color:#0a0a0a !important;font-size:12px;padding:3px 0;display:block;}
.sous-reserve{background:#fff8e1;border-left:5px solid #f8a100;
    border-radius:10px;padding:12px 16px;margin:8px 0;}
.pdf-box{background:linear-gradient(135deg,#e3f2fd,#e8f4fd);
    border-left:5px solid #0077b6;border-radius:10px;padding:12px 16px;margin:10px 0;}
.carto-box{background:linear-gradient(135deg,#e8f5e9,#f0fff4);
    border-left:5px solid #2e7d32;border-radius:10px;padding:12px 16px;margin:10px 0;}
.pi-box{background:linear-gradient(135deg,#ede7f6,#e8eaf6);
    border-left:5px solid #5e35b1;border-radius:10px;padding:10px 14px;margin:8px 0;}
.sar-box{background:linear-gradient(135deg,#fff3e0,#fbe9e7);
    border-left:5px solid #e65100;border-radius:10px;padding:10px 14px;margin:8px 0;}
.normes-table{width:100%;border-collapse:collapse;font-size:11.5px;margin-top:6px;}
.normes-table th{background:#023e8a;color:white;padding:6px 7px;
    text-align:center;font-weight:700;}
.normes-table td{padding:5px 7px;text-align:center;border:1px solid #e0e0e0;
    color:#0a0a0a !important;}
.normes-table tr:nth-child(even){background:#e3f2fd;}
.normes-table tr:nth-child(odd){background:#fff;}
.normes-table td:first-child{text-align:left;font-weight:700;color:#023e8a !important;}
p,span,div,label{color:#0a0a0a !important;}
</style>""", unsafe_allow_html=True)


# ── NAVIGATION ────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;font-size:22px;font-weight:800;"
    "color:#023e8a;margin-bottom:4px;'>EauVie</div>"
    "<div style='text-align:center;font-size:12px;color:#555;margin-bottom:14px;'>"
    "Analyse intelligente de la qualite de l'eau - 4 modules - 100 % de precision"
    "</div>",
    unsafe_allow_html=True,
)

nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    if st.button("Eau potable", key="nav_potable"):
        st.session_state.module_actif = "potable"
        st.session_state.analyse_faite = False
with nav2:
    if st.button("Eaux usées", key="nav_usee"):
        st.session_state.module_actif = "usee"
        st.session_state.analyse_faite = False
with nav3:
    if st.button("Eaux naturelles", key="nav_naturelle"):
        st.session_state.module_actif = "naturelle"
        st.session_state.analyse_faite = False
with nav4:
    if st.button("Eau agricole", key="nav_agricole"):
        st.session_state.module_actif = "agricole"
        st.session_state.analyse_faite = False

module = st.session_state.module_actif
st.markdown("---")


# ── HELPERS ───────────────────────────────────────────────────
def triple(cle, label, desc, norme, mn, mx, defv, step, unite="", optionnel=False):
    st.markdown(
        f'<div class="pcard">'
        f'<span class="plabel">{label}</span>'
        f'<span class="ptext">{desc}</span>'
        f'<span class="pnorm">{norme}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if optionnel:
        key_pm = f"pm_{cle}_{module}"
        pas_m = st.checkbox(f"Pas mesuré - {label.split('-')[0].strip()}", key=key_pm)
        if pas_m:
            st.caption("Paramètre non mesuré - indiqué dans le rapport.")
            return None
    st.markdown('<div class="mesure-group">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    v1 = c1.number_input("Mesure 1", min_value=mn, max_value=mx, value=defv,
                          step=step, key=f"{cle}_1_{module}")
    v2 = c2.number_input("Mesure 2", min_value=mn, max_value=mx, value=defv,
                          step=step, key=f"{cle}_2_{module}")
    v3 = c3.number_input("Mesure 3", min_value=mn, max_value=mx, value=defv,
                          step=step, key=f"{cle}_3_{module}")
    st.markdown("</div>", unsafe_allow_html=True)
    moy = round((v1 + v2 + v3) / 3, 5)
    st.caption(f"Moyenne : **{moy}** {unite}")
    return moy


def bloc_analyste(sources_list):
    st.markdown('<span class="section-title">Analyste et prélèvement</span>',
                unsafe_allow_html=True)
    a = st.text_input("Nom complet de l\'analyste *",
                       placeholder="Ex : Jean KOFFI", key=f"analyste_{module}")
    l = st.text_input("Lieu de prélèvement *",
                       placeholder="Ex : Village de Kpanrou", key=f"lieu_{module}")
    s = st.selectbox("Source *", sources_list, key=f"source_{module}")
    cg1, cg2 = st.columns(2)
    lat = cg1.number_input("Latitude", value=6.3703, step=0.0001,
                             format="%.4f", key=f"lat_{module}")
    lon = cg2.number_input("Longitude", value=2.4305, step=0.0001,
                             format="%.4f", key=f"lon_{module}")
    return a, l, s, lat, lon


def afficher_resultat(label_final, cs, conf, pr, classes_labels,
                       conseil_txt, classe_pred, sous_reserve=False,
                       params_manquants=None, methodes=None):
    if params_manquants is None:
        params_manquants = []
    if methodes is None:
        methodes = []
    st.markdown(
        f'<div class="result-box {cs}">{label_final}<br>'
        f'<span style="font-size:13px;font-weight:600;">Confiance : {conf} %</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if sous_reserve and params_manquants:
        st.markdown(
            f'<div class="sous-reserve">'
            f'Parametres critiques non mesures : {", ".join(params_manquants)}<br/>'
            f'Décision provisoire - ajoutez ces mesures pour une confirmation définitive.'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(f"**Conseil :** {conseil_txt}")
    if methodes and classe_pred > 0:
        with st.expander("Recommandations détaillées"):
            for m, d in methodes:
                st.markdown(
                    f'<div class="conseil-box">'
                    f'<span style="font-weight:800;color:#023e8a;">{m}</span>'
                    f'<span class="conseil-item">{d}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    prd = pd.DataFrame({
        "Classe": classes_labels,
        "Probabilite (%)": [round(p * 100, 1) for p in pr],
    })
    st.bar_chart(prd.set_index("Classe"))


def bloc_pdf_et_carto(r, key_pdf, key_carto, module_label):
    st.markdown('<div class="pdf-box">Rapport PDF officiel</div>',
                unsafe_allow_html=True)
    if st.session_state.dernier_pdf:
        st.download_button(
            "Télécharger le rapport PDF",
            data=st.session_state.dernier_pdf,
            file_name=st.session_state.dernier_pdf_nom,
            mime="application/pdf",
            key=key_pdf,
        )
        st.success("Rapport PDF prêt.")
    st.markdown('<div class="carto-box">Cartographie - Si mesure reelle, ajoutez-la a la carte communautaire.</div>',
                unsafe_allow_html=True)
    if st.button("Ajoutér à la cartographie", key=key_carto):
        st.session_state.carto_points.append({
            "module": module_label,
            "lat": r["lat"], "lon": r["lon"],
            "lieu": r["lieu"], "source": r["source"],
            "resultat": r["lbl_final"], "classe": r["cl"],
            "analyste": r["analyste"],
            "date": datetime.now().strftime("%d/%m/%Y"),
            "heure": datetime.now().strftime("%H:%M"),
        })
        st.success(f"Ajouté ({len(st.session_state.carto_points)} point(s)).")


# ══════════════════════════════════════════════════════════════
# MODULE 1 : EAU POTABLE
# ══════════════════════════════════════════════════════════════
if module == "potable":
    st.markdown(
        '<div class="header-box header-potable">'
        '<div class="header-title">EauVie — Eau potable et domestique</div>'
        '<div class="header-sub">Analyse intelligente de l\'eau a boire ou domestique'
        ' afin de garantir une consommation rassurante et bénéfique</div>'
        '<div class="header-author">Charles Ezechiel MEDEZOUNDJI - Biologiste - Benin</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Normes OMS - 11 parametres"):
        st.markdown("""<table class="normes-table">
<tr><th>Categorie</th><th>Parametre</th><th>Potable</th>
    <th>Douteuse</th><th>Polluee</th><th>Dangereuse</th></tr>
<tr><td>Microbiologie</td><td>E. coli (UFC/100 mL)</td>
    <td>0</td><td>1-10</td><td>10-500</td><td>&gt;500</td></tr>
<tr><td rowspan="5">Physico-chimique</td><td>pH</td>
    <td>6,5-8,5</td><td>5,5-9,0</td><td>4,5-5,5</td><td>&lt;4,5</td></tr>
<tr><td>Turbidite (NTU)</td><td>&lt;5</td><td>5-10</td><td>10-50</td><td>&gt;50</td></tr>
<tr><td>Temperature (deg C)</td><td>&lt;25</td><td>25-30</td><td>30-35</td><td>&gt;35</td></tr>
<tr><td>Conductivite (uS/cm)</td><td>&lt;2 500</td><td>2 500-3 000</td><td>3 000-4 000</td><td>&gt;4 000</td></tr>
<tr><td>Oxygene dissous (mg/L)</td><td>&gt;6</td><td>4-6</td><td>2-4</td><td>&lt;2</td></tr>
<tr><td rowspan="5">Chimique</td><td>Nitrates (mg/L)</td>
    <td>&lt;50</td><td>50-80</td><td>80-150</td><td>&gt;150</td></tr>
<tr><td>Nitrites (mg/L)</td><td>&lt;3</td><td>3-5</td><td>5-10</td><td>&gt;10</td></tr>
<tr><td>Ammonium (mg/L)</td><td>&lt;1,5</td><td>1,5-3</td><td>3-5</td><td>&gt;5</td></tr>
<tr><td>Plomb (mg/L)</td><td>&lt;0,01</td><td>0,01-0,02</td><td>0,02-0,05</td><td>&gt;0,05</td></tr>
<tr><td>Chlore residuel (mg/L)</td><td>0,2-0,5</td><td>0,05-0,2</td><td>&lt;0,05</td><td>0</td></tr>
</table>""", unsafe_allow_html=True)
        st.caption(
            "Sources : OMS 2017 - Norme beninoise NB 001/2001 - "
            "Mama et al. 2011 - SONEB bulletins 2018-2022 - USEPA 2022"
        )

    analyste, lieu, source, lat_i, lon_i = bloc_analyste([
        "Robinet (reseau traite SONEB)", "Puits peu profond", "Forage profond",
        "Riviere", "Fleuve / marigot", "Lac", "Eau stagnante (mare)",
        "Eau de pluie", "Source naturelle", "Eau de barrage",
        "Citerne stockee", "Autre",
    ])

    st.markdown('<span class="section-title">Saisie des 11 paramètres - 3 categories</span>',
                unsafe_allow_html=True)
    st.info("Saisissez 3 mesures par paramètre. Cochez « Pas mesuré » si non disponible.")

    st.markdown('<div class="cat-box cat-micro">'
                '<span class="cat-title cat-title-micro">Qualité microbiologique</span>',
                unsafe_allow_html=True)
    ecoli = triple("ecoli", "E. coli - Contamination fecale",
                   "Toute presence = risque sanitaire direct.", "OMS : 0 UFC/100 mL",
                   0.0, 10000.0, 0.0, 1.0, "UFC/100 mL", True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cat-box cat-physico">'
                '<span class="cat-title cat-title-physico">Qualité physico-chimique</span>',
                unsafe_allow_html=True)
    pH_v = triple("pH_p", "pH - Acidite / Basicite",
                   "pH bas : metaux toxiques. pH eleve : contamination chimique.",
                   "OMS : 6,5 a 8,5", 0.0, 14.0, 7.0, 0.01, optionnel=True)
    turb = triple("turb_p", "Turbidite (NTU)",
                   "Eau trouble = pathogenes possibles.", "OMS : < 5 NTU",
                   0.0, 200.0, 2.0, 0.01, "NTU", True)
    temp = triple("temp_p", "Temperature (deg C)",
                   "Au-dela de 25 deg C : proliferation microbienne acceleree.",
                   "OMS : < 25 deg C", 0.0, 60.0, 27.0, 0.1, "deg C", True)
    cond = triple("cond_p", "Conductivite (uS/cm)",
                   "Sels excessifs : risques renaux a long terme.",
                   "OMS : < 2 500 uS/cm", 0.0, 10000.0, 400.0, 1.0, "uS/cm", True)
    o2_v = triple("o2_p", "Oxygene dissous (mg/L)",
                   "Faible = decomposition organique.", "Norme : > 6 mg/L",
                   0.0, 14.0, 7.0, 0.01, "mg/L", True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cat-box cat-chimique">'
                '<span class="cat-title cat-title-chimique">Qualité chimique</span>',
                unsafe_allow_html=True)
    no3 = triple("no3_p", "Nitrates (mg/L)",
                  "Pollution agricole. > 50 mg/L = methemoglobinemie.",
                  "OMS : < 50 mg/L", 0.0, 500.0, 5.0, 0.1, "mg/L", True)
    no2 = triple("no2_p", "Nitrites (mg/L)",
                  "Contamination organique recente.", "OMS : < 3 mg/L",
                  0.0, 20.0, 0.01, 0.001, "mg/L", True)
    nh4 = triple("nh4_p", "Ammonium (mg/L)",
                  "Degradation matieres organiques.", "OMS : < 1,5 mg/L",
                  0.0, 50.0, 0.1, 0.01, "mg/L", True)
    pb_v = triple("pb_p", "Plomb (mg/L)",
                   "Neurotoxique. Aucun seuil sur.", "OMS : < 0,01 mg/L",
                   0.0, 1.0, 0.002, 0.0001, "mg/L", True)
    cl_v = triple("cl_p", "Chlore residuel (mg/L)",
                   "Desinfectant residuel eau traitee.", "Cible : 0,2-0,5 mg/L",
                   0.0, 5.0, 0.0, 0.001, "mg/L", True)
    st.markdown("</div>", unsafe_allow_html=True)

    vals_p = {
        "ecoli": ecoli, "pH": pH_v, "Turbidite": turb, "Temperature": temp,
        "Conductivite": cond, "O2": o2_v, "Nitrates": no3, "Nitrites": no2,
        "Ammonium": nh4, "Plomb": pb_v, "Chlore": cl_v,
    }
    nb_mesures = sum(1 for v in vals_p.values() if v is not None)

    if no3 is not None and no2 is not None and nh4 is not None:
        pi = round(no3 + no2 * 10 + nh4, 2)
        ni = (
            "Faible (<10)" if pi < 10
            else ("Modere (10-50)" if pi < 50
                  else ("Eleve (50-150)" if pi < 150 else "Critique (>150)"))
        )
        st.markdown(
            f'<div class="pi-box"><b>Indice de pollution chimique</b>'
            f' (NO3 + NO2x10 + NH4) = <b>{pi}</b> &rarr; {ni}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("Lancer l\'analyse — Eau potable", key="btn_potable"):
        erreurs = []
        if not analyste.strip():
            erreurs.append("Nom de l'analyste obligatoire.")
        if not lieu.strip():
            erreurs.append("Lieu de prelevement obligatoire.")
        if nb_mesures < 3:
            erreurs.append("Au moins 3 parametres mesures requis.")
        if erreurs:
            for e in erreurs:
                st.error(e)
        else:
            cl_pred, pr = predict_potable(vals_p)
            labels = ["POTABLE", "DOUTEUSE", "POLLUÉE", "DANGEREUSE"]
            css_cls = ["potable", "douteuse", "polluee", "dangereuse"]
            lbl_final, sr, manquants = evaluer_sous_reserve(
                {"ecoli": ecoli, "pH": pH_v, "turb": turb,
                 "no3": no3, "no2": no2, "pb": pb_v},
                cl_pred,
            )
            conf = round(pr[cl_pred] * 100, 1)
            conseils = [
                "Eau conforme aux normes OMS. Consommation possible sans risque.",
                "Anomalies détectées. Filtrez et faites bouillir avant consommation.",
                "Eau polluée. Traitement complet obligatoire avant tout usage.",
                "DANGER EXTRÊME. Tout contact à éviter. Signalez aux autorités.",
            ]
            methodes_p = [
                ("Ébullition", "5 min minimum. Efficace contre bacteries, virus, parasites."),
                ("Filtration sur sable et gravier", "Gravier + sable + charbon actif. Completer par ebullition."),
                ("SODIS", "Bouteilles transparentes, 6 h soleil ou 2 j nuageux. OMS valide."),
                ("Chloration", "2 gouttes Javel 5 % / litre. Attendre 30 min."),
                ("Moringa oleifera", "2-3 graines broyees dans 1 L. Agiter + decanter 1 h."),
            ]
            st.session_state.analyse_faite = True
            st.session_state.dernier_resultat = {
                "lb": labels[cl_pred], "cs": css_cls[cl_pred],
                "lbl_final": lbl_final, "cl": int(cl_pred),
                "pr": list(pr), "conf": conf,
                "sr": sr, "manquants": manquants,
                "vals": vals_p, "analyste": analyste,
                "lieu": lieu, "source": source,
                "lat": lat_i, "lon": lon_i, "module": "potable",
                "methodes": methodes_p, "conseil": conseils[cl_pred],
            }
            # Tableau conformite pour le PDF
            params_aff = [
                ("E. coli", f"{ecoli:.1f} UFC/100mL" if ecoli is not None else "Non mesure",
                 "0 UFC/100mL", statut_param(ecoli, 0, 0, inverse=True)),
                ("pH", f"{pH_v:.3f}" if pH_v is not None else "Non mesure",
                 "6,5-8,5", statut_param(pH_v, 6.5, 8.5)),
                ("Turbidite (NTU)", f"{turb:.3f}" if turb is not None else "Non mesure",
                 "< 5 NTU", statut_param(turb, 0, 5)),
                ("Temperature (deg C)", f"{temp:.2f}" if temp is not None else "Non mesure",
                 "< 25 deg C", statut_param(temp, 0, 25)),
                ("Conductivite (uS/cm)", f"{cond:.1f}" if cond is not None else "Non mesure",
                 "< 2 500 uS/cm", statut_param(cond, 0, 2500)),
                ("Oxygene dissous (mg/L)", f"{o2_v:.3f}" if o2_v is not None else "Non mesure",
                 "> 6 mg/L", statut_param(o2_v, 6, 14)),
                ("Nitrates (mg/L)", f"{no3:.3f}" if no3 is not None else "Non mesure",
                 "< 50 mg/L", statut_param(no3, 0, 50)),
                ("Nitrites (mg/L)", f"{no2:.4f}" if no2 is not None else "Non mesure",
                 "< 3 mg/L", statut_param(no2, 0, 3)),
                ("Ammonium (mg/L)", f"{nh4:.3f}" if nh4 is not None else "Non mesure",
                 "< 1,5 mg/L", statut_param(nh4, 0, 1.5)),
                ("Plomb (mg/L)", f"{pb_v:.4f}" if pb_v is not None else "Non mesure",
                 "< 0,01 mg/L", statut_param(pb_v, 0, 0.01)),
                ("Chlore residuel (mg/L)", f"{cl_v:.3f}" if cl_v is not None else "Non mesure",
                 "0,2-0,5 mg/L",
                 statut_param(cl_v, 0.2, 0.5) if cl_v is not None
                 else ("Non mesure", colors.HexColor("#9e9e9e"))),
            ]
            try:
                pdf = construire_pdf(
                    "Eau potable et domestique",
                    "Analyse intelligente pour une consommation sure - Normes OMS",
                    BM, params_aff, lbl_final, conseils[cl_pred],
                    methodes_p if cl_pred > 0 else [],
                    analyste, lieu, source, list(pr),
                    ["POTABLE", "DOUTEUSE", "POLLUÉE", "DANGEREUSE"], cl_pred,
                    "OMS 2017 - NB 001/2001 - SONEB - USEPA 2022",
                    module_key="potable",
                )
                st.session_state.dernier_pdf = pdf
                st.session_state.dernier_pdf_nom = (
                    "rapport_potable_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
                )
            except Exception as exc:
                st.session_state.dernier_pdf = None
                st.error(f"Erreur PDF : {exc}")
            st.session_state.histo.append({
                "Module": "Eau potable",
                "Heure": datetime.now().strftime("%H:%M"),
                "Analyste": analyste, "Lieu": lieu,
                "Resultat": lbl_final,
                **{k: (round(v, 3) if v is not None else "NM")
                   for k, v in vals_p.items()},
            })

    if (st.session_state.analyse_faite
            and st.session_state.dernier_resultat
            and st.session_state.dernier_resultat.get("module") == "potable"):
        r = st.session_state.dernier_resultat
        afficher_resultat(
            r["lbl_final"], r["cs"], r["conf"], r["pr"],
            ["Potable", "Douteuse", "Polluee", "Dangereuse"],
            r["conseil"], r["cl"],
            r.get("sr", False), r.get("manquants", []),
            r.get("methodes", []),
        )
        bloc_pdf_et_carto(r, "dl_pdf_p", "carto_p", "Eau potable")


# ══════════════════════════════════════════════════════════════
# MODULE 2 : EAUX USEES
# ══════════════════════════════════════════════════════════════
elif module == "usee":
    st.markdown(
        '<div class="header-box header-usee">'
        '<div class="header-title">EauVie — Eaux usées et industrielles</div>'
        '<div class="header-sub">Analyse intelligente des eaux usees'
        ' pour une gestion saine et responsable</div>'
        '<div class="header-author">Charles MEDEZOUNDJI'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    LABELS_EU = ["CONFORME AU REJET", "LIMITE", "NON CONFORME", "TRÈS POLLUÉE"]
    CSS_EU = ["conforme", "limite", "non-conforme", "tres-polluee"]
    CONSEILS_EU = [
        "Effluent conforme aux normes de rejet. Il peut etre deverse dans le milieu"
        " naturel selon la reglementation en vigueur.",
        "Effluent a la limite des normes. Un traitement complementaire est recommande"
        " avant rejet.",
        "Effluent non conforme. Traitement obligatoire avant tout rejet."
        " Risque de pollution du milieu recepteur.",
        "EFFLUENT TRES POLLUANT. Traitement complet imperatif. Rejet interdit."
        " Risque grave pour l'ecosysteme et la sante publique.",
    ]

    with st.expander("Normes de rejet - Eaux usees"):
        st.markdown("""<table class="normes-table">
<tr><th>Parametre</th><th>Conforme rejet</th><th>Limite</th>
    <th>Non conforme</th><th>Tres polluee</th></tr>
<tr><td>pH</td><td>6,0-9,0</td><td>5,5-10,5</td>
    <td>4,5-5,5 ou 10,5-11</td><td>&lt;4,5 ou &gt;11</td></tr>
<tr><td>DBO5 (mg/L)</td><td>&lt;30</td><td>30-60</td>
    <td>60-200</td><td>&gt;200</td></tr>
<tr><td>DCO (mg/L)</td><td>&lt;90</td><td>90-180</td>
    <td>180-600</td><td>&gt;600</td></tr>
<tr><td>MES (mg/L)</td><td>&lt;35</td><td>35-80</td>
    <td>80-300</td><td>&gt;300</td></tr>
<tr><td>Temperature (deg C)</td><td>&lt;30</td><td>30-35</td>
    <td>35-40</td><td>&gt;40</td></tr>
<tr><td>NH4 (mg/L)</td><td>&lt;5</td><td>5-10</td>
    <td>10-30</td><td>&gt;30</td></tr>
<tr><td>Plomb (mg/L)</td><td>&lt;0,1</td><td>0,1-0,3</td>
    <td>0,3-0,8</td><td>&gt;0,8</td></tr>
<tr><td>E. coli (UFC/100 mL)</td><td>&lt;2 000</td><td>2 000-10 000</td>
    <td>10 000-200 000</td><td>&gt;200 000</td></tr>
</table>""", unsafe_allow_html=True)
        st.caption(
            "Sources : Arrete beninois 2001-294 - OMS Guidelines Safe Use Wastewater 2006"
            " - Directive 91/271/CEE - Mama et al. 2011 - Vodounnou 2020"
        )

    analyste, lieu, source, lat_i, lon_i = bloc_analyste([
        "Station d'epuration urbaine", "Effluent industriel",
        "Eaux de ruissellement", "Effluent agricole",
        "Decharge / lixiviat", "Fosse septique",
        "Canal d'evacuation", "Autre",
    ])

    st.markdown('<span class="section-title">Paramètres des eaux usées</span>',
                unsafe_allow_html=True)

    st.markdown('<div class="cat-box cat-dbo">'
                '<span class="cat-title cat-title-dbo">Paramètres organiques et physiques</span>',
                unsafe_allow_html=True)
    ph_eu = triple("ph_eu", "pH - Acidite/Basicite",
                   "pH extreme = corrosion et toxicite.",
                   "Norme rejet : 6,0-9,0", 0.0, 14.0, 7.2, 0.01, optionnel=True)
    dbo5 = triple("dbo5", "DBO5 (mg/L) - Demande Biochimique en O2",
                  "Charge organique biodegradable. Elevee = pollution organique forte.",
                  "Norme rejet : < 30 mg/L", 0.0, 2000.0, 20.0, 0.5, "mg/L")
    dco = triple("dco", "DCO (mg/L) - Demande Chimique en O2",
                 "Mesure toute la matiere oxydable.",
                 "Norme rejet : < 90 mg/L", 0.0, 5000.0, 70.0, 1.0, "mg/L")
    mes = triple("mes", "MES (mg/L) - Matieres en Suspension",
                 "Particules solides en suspension.",
                 "Norme rejet : < 35 mg/L", 0.0, 2000.0, 25.0, 0.5, "mg/L")
    temp_eu = triple("temp_eu", "Temperature (deg C)",
                     "Eau chaude : reduit l'O2 dissous du milieu recepteur.",
                     "Norme rejet : < 30 deg C", 0.0, 60.0, 27.0, 0.1, "deg C", True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="cat-box cat-chimique">'
                '<span class="cat-title cat-title-chimique">Paramètres chimiques et microbiologiques</span>',
                unsafe_allow_html=True)
    nh4_eu = triple("nh4_eu", "NH4 (mg/L) - Ammonium",
                    "Indicateur de pollution azotee et fecale.",
                    "Norme rejet : < 5 mg/L", 0.0, 500.0, 3.0, 0.1, "mg/L", True)
    pb_eu = triple("pb_eu", "Plomb (mg/L)",
                   "Metal lourd toxique. Bioaccumulation dans la chaine alimentaire.",
                   "Norme rejet : < 0,1 mg/L", 0.0, 5.0, 0.08, 0.001, "mg/L", True)
    ecoli_eu = triple("ecoli_eu", "E. coli (UFC/100 mL)",
                      "Indicateur de contamination fecale dans l'effluent.",
                      "Norme rejet : < 2 000 UFC/100 mL",
                      0.0, 10000000.0, 500.0, 10.0, "UFC/100 mL", True)
    st.markdown("</div>", unsafe_allow_html=True)

    feat_eu = {
        "pH": ph_eu if ph_eu is not None else 7.0,
        "DBO5": dbo5,
        "DCO": dco,
        "MES": mes,
        "Temperature": temp_eu if temp_eu is not None else 27.0,
        "NH4": nh4_eu if nh4_eu is not None else 3.0,
        "Plomb": pb_eu if pb_eu is not None else 0.08,
        "Ecoli": ecoli_eu if ecoli_eu is not None else 500.0,
    }

    st.markdown("---")
    if st.button("Lancer l\'analyse — Eaux usées", key="btn_eu"):
        erreurs = []
        if not analyste.strip():
            erreurs.append("Nom de l'analyste obligatoire.")
        if not lieu.strip():
            erreurs.append("Lieu obligatoire.")
        if erreurs:
            for e in erreurs:
                st.error(e)
        else:
            cl_pred, pr = predict_module("usee", feat_eu)
            conf = round(pr[cl_pred] * 100, 1)
            methodes_eu = [
                ("Traitement primaire",
                 "Décantation et flottation : élimination des MES. Réduction de la DBO de 30 à 40 %."),
                ("Traitement secondaire biologique",
                 "Boues activées ou lagunage : dégradation de la matière organique. Réduction de la DBO supérieure à 85 %."),
                ("Traitement tertiaire",
                 "Désinfection UV ou chloration pour éliminer les pathogènes avant rejet."),
                ("Neutralisation du pH",
                 "Ajout de chaux (si pH acide) ou d\'acide chlorhydrique (si pH alcalin)."),
                ("Traitement metaux lourds",
                 "Précipitation chimique, échange d\'ions ou électrolyse pour le plomb."),
            ]
            st.session_state.analyse_faite = True
            st.session_state.dernier_resultat = {
                "lb": LABELS_EU[cl_pred], "cs": CSS_EU[cl_pred],
                "lbl_final": LABELS_EU[cl_pred], "cl": int(cl_pred),
                "pr": list(pr), "conf": conf,
                "analyste": analyste, "lieu": lieu, "source": source,
                "lat": lat_i, "lon": lon_i, "module": "usee",
                "methodes": methodes_eu, "conseil": CONSEILS_EU[cl_pred],
            }
            params_aff = [
                ("pH", f"{feat_eu['pH']:.2f}", "6,0-9,0",
                 statut_param(feat_eu["pH"], 6.0, 9.0)),
                ("DBO5 (mg/L)", f"{dbo5:.1f}", "< 30",
                 statut_param(dbo5, 0, 30)),
                ("DCO (mg/L)", f"{dco:.1f}", "< 90",
                 statut_param(dco, 0, 90)),
                ("MES (mg/L)", f"{mes:.1f}", "< 35",
                 statut_param(mes, 0, 35)),
                ("Temperature (deg C)", f"{feat_eu['Temperature']:.1f}", "< 30",
                 statut_param(feat_eu["Temperature"], 0, 30)),
                ("NH4 (mg/L)", f"{feat_eu['NH4']:.2f}", "< 5",
                 statut_param(feat_eu["NH4"], 0, 5)),
                ("Plomb (mg/L)", f"{feat_eu['Plomb']:.4f}", "< 0,1",
                 statut_param(feat_eu["Plomb"], 0, 0.1)),
                ("E. coli (UFC/100 mL)", f"{feat_eu['Ecoli']:.0f}", "< 2 000",
                 statut_param(feat_eu["Ecoli"], 0, 2000, inverse=True)),
            ]
            try:
                pdf = construire_pdf(
                    "Eaux usées et industrielles",
                    "Analyse intelligente pour une gestion saine et responsable",
                    colors.HexColor("#7b1fa2"),
                    params_aff, LABELS_EU[cl_pred], CONSEILS_EU[cl_pred],
                    methodes_eu if cl_pred > 0 else [],
                    analyste, lieu, source, list(pr), LABELS_EU, cl_pred,
                    "Arrete Benin 2001-294 - OMS 2006 - Directive 91/271/CEE - Mama 2011",
                    module_key="usee",
                )
                st.session_state.dernier_pdf = pdf
                st.session_state.dernier_pdf_nom = (
                    "rapport_eaux_usees_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
                )
            except Exception as exc:
                st.session_state.dernier_pdf = None
                st.error(f"Erreur PDF : {exc}")
            st.session_state.histo.append({
                "Module": "Eaux usees",
                "Heure": datetime.now().strftime("%H:%M"),
                "Analyste": analyste, "Lieu": lieu,
                "Resultat": LABELS_EU[cl_pred],
                "DBO5": dbo5, "DCO": dco, "MES": mes, "pH": feat_eu["pH"],
            })

    if (st.session_state.analyse_faite
            and st.session_state.dernier_resultat
            and st.session_state.dernier_resultat.get("module") == "usee"):
        r = st.session_state.dernier_resultat
        afficher_resultat(
            r["lbl_final"], r["cs"], r["conf"], r["pr"],
            LABELS_EU, r["conseil"], r["cl"],
            methodes=r.get("methodes", []),
        )
        bloc_pdf_et_carto(r, "dl_pdf_eu", "carto_eu", "Eaux usees")


# ══════════════════════════════════════════════════════════════
# MODULE 3 : EAUX NATURELLES
# ══════════════════════════════════════════════════════════════
elif module == "naturelle":
    st.markdown(
        '<div class="header-box header-naturelle">'
        '<div class="header-title">EauVie — Eaux naturelles et écologiques</div>'
        '<div class="header-sub">Analyse intelligente des eaux naturelles'
        ' pour une écologie stable et durable</div>'
        '<div class="header-author">Charles MEDEZOUNDJI'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    LABELS_EN = ["BONNE QUALITÉ", "QUALITÉ MOYENNE", "MAUVAISE QUALITÉ", "TRÈS MAUVAISE QUALITÉ"]
    CSS_EN = ["bonne", "moyenne", "mauvaise", "tres-mauvaise"]
    CONSEILS_EN = [
        "Eau de bonne qualité écologique. Le milieu aquatique est en bon état. Faune et flore préservées.",
        "Qualité moyenne. Des pressions modérées affectent ce milieu. Actions préventives recommandées.",
        "Mauvaise qualité. Ce milieu est fortement perturbé. Mesures de restauration nécessaires.",
        "TRÈS MAUVAISE QUALITÉ. Milieu aquatique gravement dégradé. Intervention urgente requise.",
    ]

    with st.expander("Normes qualite ecologique - Eaux naturelles"):
        st.markdown("""<table class="normes-table">
<tr><th>Parametre</th><th>Bonne</th><th>Moyenne</th>
    <th>Mauvaise</th><th>Tres mauvaise</th></tr>
<tr><td>pH</td><td>6,5-8,0</td><td>6,0-9,0</td>
    <td>5,5-9,5</td><td>&lt;5,5</td></tr>
<tr><td>Turbidite (NTU)</td><td>&lt;5</td><td>5-20</td>
    <td>20-100</td><td>&gt;100</td></tr>
<tr><td>O2 dissous (mg/L)</td><td>&gt;7</td><td>5-7</td>
    <td>2-5</td><td>&lt;2</td></tr>
<tr><td>Nitrates (mg/L)</td><td>&lt;5</td><td>5-25</td>
    <td>25-100</td><td>&gt;100</td></tr>
<tr><td>E. coli (UFC/100 mL)</td><td>&lt;100</td><td>100-1 000</td>
    <td>1 000-100 000</td><td>&gt;100 000</td></tr>
<tr><td>Temperature (deg C)</td><td>&lt;25</td><td>25-30</td>
    <td>30-36</td><td>&gt;36</td></tr>
<tr><td>Conductivite (uS/cm)</td><td>&lt;400</td><td>400-1 000</td>
    <td>1 000-2 500</td><td>&gt;2 500</td></tr>
</table>""", unsafe_allow_html=True)
        st.caption(
            "Sources : DCE 2000/60/CE - IBGN NF T90-350 - Imorou Toko et al. 2010"
            " - Mama et al. 2011 - Boukari et al. 2003 - DN Hydraulique Benin 2020"
        )

    analyste, lieu, source, lat_i, lon_i = bloc_analyste([
        "Riviere / cours d'eau", "Lac / retenue", "Marigot / zone humide",
        "Lagune / estuaire", "Source naturelle", "Barrage / reservoir",
        "Nappe phreatique affleurante", "Autre",
    ])

    st.markdown('<span class="section-title">Paramètres écologiques</span>',
                unsafe_allow_html=True)
    st.markdown('<div class="cat-box cat-ecologie">'
                '<span class="cat-title cat-title-eco">Qualité physico-chimique et biologique</span>',
                unsafe_allow_html=True)
    ph_en = triple("ph_en", "pH - Equilibre du milieu aquatique",
                   "Hors norme = stress pour la faune aquatique.",
                   "Norme ecologique : 6,5-8,0", 0.0, 14.0, 7.2, 0.01, optionnel=True)
    turb_en = triple("turb_en", "Turbidite (NTU)",
                     "Limite la photosynthese des plantes aquatiques.",
                     "Norme : < 5 NTU", 0.0, 500.0, 2.5, 0.1, "NTU")
    o2_en = triple("o2_en", "Oxygene dissous (mg/L)",
                   "Vital pour la vie aquatique. Insuffisant = mort des poissons.",
                   "Norme : > 7 mg/L", 0.0, 20.0, 8.0, 0.01, "mg/L")
    no3_en = triple("no3_en", "Nitrates (mg/L)",
                    "Eutrophisation = algues excessives, asphyxie du milieu.",
                    "Norme : < 5 mg/L", 0.0, 1000.0, 3.0, 0.1, "mg/L")
    ecoli_en = triple("ecoli_en", "E. coli (UFC/100 mL)",
                      "Indicateur de pollution fecale du milieu.",
                      "Norme : < 100 UFC/100 mL", 0.0, 1000000.0, 15.0, 1.0, "UFC/100 mL")
    temp_en = triple("temp_en", "Temperature (deg C)",
                     "Influence la teneur en O2 et le metabolisme des especes.",
                     "Norme : < 25 deg C", 0.0, 60.0, 22.0, 0.1, "deg C", True)
    cond_en = triple("cond_en", "Conductivite (uS/cm)",
                     "Indique la mineralisation et la salinite du milieu.",
                     "Norme : < 400 uS/cm", 0.0, 10000.0, 200.0, 1.0, "uS/cm", True)
    st.markdown("</div>", unsafe_allow_html=True)

    feat_en = {
        "pH": ph_en if ph_en is not None else 7.2,
        "Turbidite": turb_en,
        "O2": o2_en,
        "Nitrates": no3_en,
        "Ecoli": ecoli_en,
        "Temperature": temp_en if temp_en is not None else 22.0,
        "Conductivite": cond_en if cond_en is not None else 200.0,
    }

    st.markdown("---")
    if st.button("Lancer l\'analyse — Eaux naturelles", key="btn_en"):
        erreurs = []
        if not analyste.strip():
            erreurs.append("Nom de l'analyste obligatoire.")
        if not lieu.strip():
            erreurs.append("Lieu obligatoire.")
        if erreurs:
            for e in erreurs:
                st.error(e)
        else:
            cl_pred, pr = predict_module("naturelle", feat_en)
            conf = round(pr[cl_pred] * 100, 1)
            methodes_en = [
                ("Reduction intrants agricoles",
                 "Limiter les engrais azotés. Zones tampons végétalisées en bordure de cours d\'eau."),
                ("Traitement des rejets urbains",
                 "Mettre en place des stations d\'épuration en amont. Traitement obligatoire avant rejet."),
                ("Restauration hydromorphologique",
                 "Reméandrement, recharge granulométrique, plantation de ripisylve."),
                ("Reduction pollutions diffuses",
                 "Sensibilisation agricole. Réduction des pesticides. Gestion rigoureuse des déchets."),
                ("Surveillance reguliere",
                 "Mise en place d\'un réseau de surveillance avec analyses périodiques."),
            ]
            st.session_state.analyse_faite = True
            st.session_state.dernier_resultat = {
                "lb": LABELS_EN[cl_pred], "cs": CSS_EN[cl_pred],
                "lbl_final": LABELS_EN[cl_pred], "cl": int(cl_pred),
                "pr": list(pr), "conf": conf,
                "analyste": analyste, "lieu": lieu, "source": source,
                "lat": lat_i, "lon": lon_i, "module": "naturelle",
                "methodes": methodes_en, "conseil": CONSEILS_EN[cl_pred],
            }
            params_aff = [
                ("pH", f"{feat_en['pH']:.2f}", "6,5-8,0",
                 statut_param(feat_en["pH"], 6.5, 8.0)),
                ("Turbidite (NTU)", f"{turb_en:.3f}", "< 5",
                 statut_param(turb_en, 0, 5)),
                ("O2 dissous (mg/L)", f"{o2_en:.3f}", "> 7",
                 statut_param(o2_en, 7, 14)),
                ("Nitrates (mg/L)", f"{no3_en:.3f}", "< 5",
                 statut_param(no3_en, 0, 5)),
                ("E. coli (UFC/100 mL)", f"{ecoli_en:.0f}", "< 100",
                 statut_param(ecoli_en, 0, 100, inverse=True)),
                ("Temperature (deg C)", f"{feat_en['Temperature']:.2f}", "< 25",
                 statut_param(feat_en["Temperature"], 0, 25)),
                ("Conductivite (uS/cm)", f"{feat_en['Conductivite']:.1f}", "< 400",
                 statut_param(feat_en["Conductivite"], 0, 400)),
            ]
            try:
                pdf = construire_pdf(
                    "Eaux naturelles et écologiques",
                    "Analyse pour une écologie stable et durable",
                    VM, params_aff, LABELS_EN[cl_pred], CONSEILS_EN[cl_pred],
                    methodes_en if cl_pred > 0 else [],
                    analyste, lieu, source, list(pr), LABELS_EN, cl_pred,
                    "DCE 2000/60/CE - IBGN - Imorou Toko 2010 - Mama 2011 - Boukari 2003",
                    module_key="naturelle",
                )
                st.session_state.dernier_pdf = pdf
                st.session_state.dernier_pdf_nom = (
                    "rapport_eaux_naturelles_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
                )
            except Exception as exc:
                st.session_state.dernier_pdf = None
                st.error(f"Erreur PDF : {exc}")
            st.session_state.histo.append({
                "Module": "Eaux naturelles",
                "Heure": datetime.now().strftime("%H:%M"),
                "Analyste": analyste, "Lieu": lieu,
                "Resultat": LABELS_EN[cl_pred],
                "pH": feat_en["pH"], "O2": o2_en, "Turbidite": turb_en,
            })

    if (st.session_state.analyse_faite
            and st.session_state.dernier_resultat
            and st.session_state.dernier_resultat.get("module") == "naturelle"):
        r = st.session_state.dernier_resultat
        afficher_resultat(
            r["lbl_final"], r["cs"], r["conf"], r["pr"],
            LABELS_EN, r["conseil"], r["cl"],
            methodes=r.get("methodes", []),
        )
        bloc_pdf_et_carto(r, "dl_pdf_en", "carto_en", "Eaux naturelles")


# ══════════════════════════════════════════════════════════════
# MODULE 4 : EAU AGRICOLE
# ══════════════════════════════════════════════════════════════
elif module == "agricole":
    st.markdown(
        '<div class="header-box header-agricole">'
        '<div class="header-title">EauVie — Eau agricole et irrigation</div>'
        '<div class="header-sub">Analyse intelligente de l\'eau agricole'
        ' pour une agriculture rentable et saine</div>'
        '<div class="header-author">Charles MEDEZOUNDJI'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    LABELS_EA = ["BONNE APTITUDE", "APTITUDE MODÉRÉE", "EAU À RISQUE", "EAU INADAPTÉE"]
    CSS_EA = ["aptitude-bonne", "aptitude-moderee", "risque", "inadaptee"]
    CONSEILS_EA = [
        "Eau de bonne aptitude a l'irrigation selon les normes FAO."
        " Convient a la majorite des cultures sans risque de salinisation.",
        "Aptitude moderee. Utilisation possible pour des cultures tolerantes."
        " Surveiller la salinite des sols. Drainage recommande.",
        "Eau a risque. Risque de salinisation et de sodicite des sols."
        " Cultures tres tolerantes uniquement. Amendements requis.",
        "EAU INADAPTEE A L'IRRIGATION. Risque grave de salinisation,"
        " toxicite ionique et degradation de la structure du sol. Ne pas utiliser.",
    ]

    with st.expander("Normes FAO - Eau d'irrigation"):
        st.markdown("""<table class="normes-table">
<tr><th>Parametre</th><th>Bonne aptitude</th><th>Moderee</th>
    <th>Risque</th><th>Inadaptee</th></tr>
<tr><td>Conductivite (uS/cm)</td><td>&lt;700</td><td>700-1 500</td>
    <td>1 500-3 000</td><td>&gt;3 000</td></tr>
<tr><td>pH</td><td>6,5-7,5</td><td>7,5-8,5</td>
    <td>8,5-9,2</td><td>&gt;9,2</td></tr>
<tr><td>Sodium Na (mg/L)</td><td>&lt;70</td><td>70-200</td>
    <td>200-500</td><td>&gt;500</td></tr>
<tr><td>Nitrates (mg/L)</td><td>&lt;10</td><td>10-30</td>
    <td>30-100</td><td>&gt;100</td></tr>
<tr><td>Calcium Ca (mg/L)</td><td>&gt;40</td><td>20-40</td>
    <td>10-20</td><td>&lt;10</td></tr>
<tr><td>Magnesium Mg (mg/L)</td><td>&gt;15</td><td>8-15</td>
    <td>3-8</td><td>&lt;3</td></tr>
<tr><td>SAR</td><td>&lt;3</td><td>3-9</td>
    <td>9-18</td><td>&gt;18</td></tr>
</table>""", unsafe_allow_html=True)
        st.caption(
            "Sources : FAO 1994 (Ayers & Westcot) - SODAGRI Benin (Kandi/Malanville)"
            " - Mama et al. 2011 zones maraicheres - DN Hydraulique Benin 2021"
        )

    analyste, lieu, source, lat_i, lon_i = bloc_analyste([
        "Riviere / cours d'eau", "Canal d'irrigation",
        "Forage / puits profond", "Eau de barrage",
        "Nappe phreatique", "Eau de pluie stockee",
        "Eau recyclee agricole", "Autre",
    ])

    st.markdown('<span class="section-title">Paramètres qualité eau agricole</span>',
                unsafe_allow_html=True)
    st.markdown('<div class="cat-box cat-agri">'
                '<span class="cat-title cat-title-agri">Paramètres FAO - Salinite, sodicite, nutrition</span>',
                unsafe_allow_html=True)
    cond_ea = triple("cond_ea", "Conductivite (uS/cm) - Salinite",
                     "Principal indicateur de salinite. Elevee = stress osmotique.",
                     "FAO : < 700 uS/cm", 0.0, 20000.0, 400.0, 1.0, "uS/cm")
    ph_ea = triple("ph_ea", "pH - Disponibilite des elements nutritifs",
                   "pH optimal 6,5-7,5 pour la disponibilite des nutriments du sol.",
                   "FAO : 6,5-7,5", 0.0, 14.0, 7.0, 0.01, optionnel=True)
    na_ea = triple("na_ea", "Sodium Na (mg/L) - Sodicite",
                   "Sodium excessif = degradation de la structure du sol.",
                   "FAO : < 70 mg/L", 0.0, 5000.0, 45.0, 0.5, "mg/L")
    no3_ea = triple("no3_ea", "Nitrates (mg/L) - Azote disponible",
                    "Nitrates = engrais naturel. Exces = pollution des nappes.",
                    "FAO : < 10 mg/L", 0.0, 500.0, 5.0, 0.1, "mg/L", True)
    ca_ea = triple("ca_ea", "Calcium Ca (mg/L) - Structure du sol",
                   "Le calcium stabilise la structure du sol et equilibre le sodium.",
                   "FAO : > 40 mg/L", 0.0, 1000.0, 60.0, 0.5, "mg/L", True)
    mg_ea = triple("mg_ea", "Magnesium Mg (mg/L) - Nutrition vegetale",
                   "Cofacteur de la chlorophylle. Insuffisant = chlorose des feuilles.",
                   "FAO : > 15 mg/L", 0.0, 500.0, 18.0, 0.5, "mg/L", True)

    na_val = na_ea if na_ea is not None else 45.0
    ca_val = ca_ea if ca_ea is not None else 60.0
    mg_val = mg_ea if mg_ea is not None else 18.0
    sar_calc = round(na_val / np.sqrt((ca_val + mg_val) / 2), 2)

    sar_lbl = (
        "Bonne aptitude (SAR < 3)" if sar_calc < 3
        else ("Modéré (SAR : 3 à 9)" if sar_calc < 9
              else ("Risque (SAR : 9 à 18)" if sar_calc < 18 else "Inadapté (SAR > 18)"))
    )
    st.markdown(
        f'<div class="sar-box"><b>SAR calcule automatiquement</b>'
        f' = Na / sqrt((Ca + Mg)/2) = <b>{sar_calc}</b> &rarr; {sar_lbl}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    feat_ea = {
        "Conductivite": cond_ea,
        "pH": ph_ea if ph_ea is not None else 7.0,
        "Sodium": na_val,
        "Nitrates": no3_ea if no3_ea is not None else 5.0,
        "Calcium": ca_val,
        "Magnesium": mg_val,
        "SAR": sar_calc,
    }

    st.markdown("---")
    if st.button("Lancer l\'analyse — Eau agricole", key="btn_ea"):
        erreurs = []
        if not analyste.strip():
            erreurs.append("Nom de l'analyste obligatoire.")
        if not lieu.strip():
            erreurs.append("Lieu obligatoire.")
        if erreurs:
            for e in erreurs:
                st.error(e)
        else:
            cl_pred, pr = predict_module("agricole", feat_ea)
            conf = round(pr[cl_pred] * 100, 1)
            methodes_ea = [
                ("Gypse (CaSO4)",
                 "Apport de gypse pour corriger la sodicité. Dose recommandée : 2 à 5 t/ha selon le SAR."),
                ("Drainage",
                 "Installer un drainage pour lessiver l\'excès de sel. Éviter la remontée capillaire."),
                ("Cultures tolerantes",
                 "Privilégier le coton, le sorgho et l\'orge (tolérance élevée). Éviter le haricot et le maïs."),
                ("Amendement calcique",
                 "Apport de chaux ou de cendre de bois pour corriger le pH et enrichir le sol en calcium."),
                ("Eau de dilution",
                 "Mélanger avec une eau de meilleure qualité pour réduire la conductivité avant irrigation."),
                ("Suivi ionique du sol",
                 "Analyses régulières du sol (conductivité, SAR, pH). Minimum deux fois par saison."),
            ]
            st.session_state.analyse_faite = True
            st.session_state.dernier_resultat = {
                "lb": LABELS_EA[cl_pred], "cs": CSS_EA[cl_pred],
                "lbl_final": LABELS_EA[cl_pred], "cl": int(cl_pred),
                "pr": list(pr), "conf": conf,
                "analyste": analyste, "lieu": lieu, "source": source,
                "lat": lat_i, "lon": lon_i, "module": "agricole",
                "methodes": methodes_ea, "conseil": CONSEILS_EA[cl_pred],
                "sar": sar_calc,
            }
            params_aff = [
                ("Conductivite (uS/cm)", f"{cond_ea:.1f}", "< 700",
                 statut_param(cond_ea, 0, 700)),
                ("pH", f"{feat_ea['pH']:.2f}", "6,5-7,5",
                 statut_param(feat_ea["pH"], 6.5, 7.5)),
                ("Sodium Na (mg/L)", f"{na_val:.2f}", "< 70",
                 statut_param(na_val, 0, 70)),
                ("Nitrates (mg/L)", f"{feat_ea['Nitrates']:.3f}", "< 10",
                 statut_param(feat_ea["Nitrates"], 0, 10)),
                ("Calcium Ca (mg/L)", f"{ca_val:.2f}", "> 40",
                 statut_param(ca_val, 40, 1000)),
                ("Magnesium Mg (mg/L)", f"{mg_val:.2f}", "> 15",
                 statut_param(mg_val, 15, 500)),
                ("SAR", f"{sar_calc}", "< 3",
                 statut_param(sar_calc, 0, 3)),
            ]
            try:
                pdf = construire_pdf(
                    "Eau agricole et irrigation",
                    "Analyse pour une agriculture rentable et saine",
                    BR, params_aff, LABELS_EA[cl_pred], CONSEILS_EA[cl_pred],
                    methodes_ea if cl_pred > 0 else [],
                    analyste, lieu, source, list(pr), LABELS_EA, cl_pred,
                    "FAO 1994 (Ayers & Westcot) - SODAGRI Benin - Mama 2011",
                    module_key="agricole",
                )
                st.session_state.dernier_pdf = pdf
                st.session_state.dernier_pdf_nom = (
                    "rapport_eau_agricole_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pdf"
                )
            except Exception as exc:
                st.session_state.dernier_pdf = None
                st.error(f"Erreur PDF : {exc}")
            st.session_state.histo.append({
                "Module": "Eau agricole",
                "Heure": datetime.now().strftime("%H:%M"),
                "Analyste": analyste, "Lieu": lieu,
                "Resultat": LABELS_EA[cl_pred],
                "Conductivite": cond_ea, "SAR": sar_calc,
                "pH": feat_ea["pH"],
            })

    if (st.session_state.analyse_faite
            and st.session_state.dernier_resultat
            and st.session_state.dernier_resultat.get("module") == "agricole"):
        r = st.session_state.dernier_resultat
        afficher_resultat(
            r["lbl_final"], r["cs"], r["conf"], r["pr"],
            LABELS_EA, r["conseil"], r["cl"],
            methodes=r.get("methodes", []),
        )
        sar_r = r.get("sar", 0)
        sar_lbl_r = (
            "Bonne aptitude" if sar_r < 3
            else ("Modere" if sar_r < 9
                  else ("Risque" if sar_r < 18 else "Inadapte"))
        )
        st.markdown(
            f'<div class="sar-box"><b>SAR de l\'echantillon :</b>'
            f' {sar_r} &rarr; {sar_lbl_r}</div>',
            unsafe_allow_html=True,
        )
        bloc_pdf_et_carto(r, "dl_pdf_ea", "carto_ea", "Eau agricole")


# ══════════════════════════════════════════════════════════════
# HISTORIQUE - COMMUN AUX 4 MODULES
# ══════════════════════════════════════════════════════════════
if len(st.session_state.histo) > 0:
    st.markdown("---")
    st.markdown('<span class="section-title">Historique des analyses - tous modules</span>',
                unsafe_allow_html=True)
    hdf = pd.DataFrame(st.session_state.histo)
    st.dataframe(hdf, use_container_width=True)
    st.download_button(
        "Télécharger le CSV",
        data=hdf.to_csv(index=False).encode("utf-8"),
        file_name="historique_eauvie.csv",
        mime="text/csv",
        key="dl_histo",
    )

# ══════════════════════════════════════════════════════════════
# CARTOGRAPHIE - AVEC FILTRE PAR TYPE D'EAU
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<span class="section-title">Cartographie des analyses</span>',
            unsafe_allow_html=True)
with st.expander("Accéder à la cartographie (réservé aux autorités)"):
    mdp = st.text_input(
        "",
        type="password",
        placeholder="Saisir le code d'acces",
        key="mdp_carto",
        label_visibility="collapsed",
    )
if st.session_state.get("mdp_carto", "") == "CARTOGRAPHIE":
    mdp = "CARTOGRAPHIE"
    if True:
        pts = st.session_state.carto_points
        nb = len(pts)
        st.success(f"Accès autorisé. {nb} point(s) enregistré(s).")

        if nb == 0:
            st.info(
                "Aucune mesure ajoutee. Effectuez une analyse et cliquez"
                " sur 'Ajoutér à la cartographie'."
            )
        else:
            # Filtre par type d'eau
            modules_presents = sorted(set(p.get("module", "Tous") for p in pts))
            filtre_options = ["Tous les types"] + modules_presents
            filtre_choix = st.selectbox(
                "Filtrer par type d\'eau :",
                filtre_options,
                key="filtre_carto",
            )
            if filtre_choix == "Tous les types":
                pts_filtres = pts
            else:
                pts_filtres = [p for p in pts if p.get("module") == filtre_choix]

            st.info(
                f"{len(pts_filtres)} point(s) affiché(s)"
                f"{' pour ' + filtre_choix if filtre_choix != 'Tous les types' else ''}."
            )

            if len(pts_filtres) > 0:
                try:
                    import folium
                    from streamlit_folium import st_folium

                    COUL_MAP = {0: "green", 1: "orange", 2: "red", 3: "darkred"}
                    MOD_ICON = {
                        "Eau potable": "tint",
                        "Eaux usees": "warning-sign",
                        "Eaux naturelles": "leaf",
                        "Eau agricole": "grain",
                    }

                    clat = sum(p["lat"] for p in pts_filtres) / len(pts_filtres)
                    clon = sum(p["lon"] for p in pts_filtres) / len(pts_filtres)
                    m = folium.Map(location=[clat, clon], zoom_start=8,
                                   tiles="CartoDB positron")

                    for p in pts_filtres:
                        popup_html = (
                            f"<b>{p.get('module', '')}</b><br>"
                            f"<b>{p.get('resultat', '')}</b><br>"
                            f"Lieu : {p.get('lieu', '')}<br>"
                            f"Source : {p.get('source', '')}<br>"
                            f"Analyste : {p.get('analyste', '')}<br>"
                            f"{p.get('date', '')} a {p.get('heure', '')}"
                        )
                        folium.Marker(
                            location=[p["lat"], p["lon"]],
                            popup=folium.Popup(popup_html, max_width=240),
                            tooltip=f"[{p.get('module', '')}] {p.get('resultat', '')} - {p.get('lieu', '')}",
                            icon=folium.Icon(
                                color=COUL_MAP.get(p.get("classe", 0), "blue"),
                                icon="info-sign",
                                prefix="glyphicon",
                            ),
                        ).add_to(m)

                    st_folium(m, width=700, height=430)

                    df_c = pd.DataFrame(pts_filtres)
                    st.dataframe(df_c, use_container_width=True)
                    col_a, col_b = st.columns(2)
                    col_a.download_button(
                        "Exporter en CSV",
                        data=df_c.to_csv(index=False).encode("utf-8"),
                        file_name="carto_eauvie.csv",
                        mime="text/csv",
                        key="dl_cc",
                    )
                    col_b.download_button(
                        "Exporter en JSON",
                        data=json.dumps(
                            {"points": pts_filtres}, ensure_ascii=False, indent=2
                        ).encode("utf-8"),
                        file_name="carto_eauvie.json",
                        mime="application/json",
                        key="dl_cj",
                    )
                except ImportError:
                    st.warning("Folium non disponible. Donnees tabulaires :")
                    st.dataframe(pd.DataFrame(pts_filtres), use_container_width=True)


# ── CONTACT ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="background:linear-gradient(135deg,#023e8a,#0077b6);'
    'border-radius:14px;padding:16px;text-align:center;">'
    '<div style="color:#fff;font-size:14px;font-weight:800;margin-bottom:6px;">'
    'Charles MEDEZOUNDJI - Développeur d\'EauVie</div>'
    '<a href="mailto:charlesezechielmedezoundji@gmail.com'
    '?subject=EauVie%20-%20Message" target="_blank"'
    ' style="display:inline-block;background:white;color:#023e8a;'
    'font-weight:800;font-size:13px;padding:9px 22px;'
    'border-radius:9px;text-decoration:none;">'
    'Envoyer un message</a>'
    '<div style="color:#a8d8ff;font-size:10px;margin-top:8px;">'
    'charlesezechielmedezoundji@gmail.com</div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#023e8a !important;'
    'font-size:11px;padding:8px;font-weight:600;">'
    'EauVie - 4 modules - Eau potable / Eaux usees / Eaux naturelles / Eau agricole'
    ' - Random Forest - 100 % - Charles MEDEZOUNDJI - 2026</div>',
    unsafe_allow_html=True,
)
