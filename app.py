
import streamlit as st
import pandas as pd
import numpy as np
import io, json
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

# ── SESSION STATE ─────────────────────────────────────────────
for k,v in [("carto_points",[]),("dernier_pdf",None),("dernier_pdf_nom",""),
            ("histo",[]),("analyse_faite",False),("dernier_resultat",None)]:
    if k not in st.session_state: st.session_state[k]=v

# ── COULEURS PDF ──────────────────────────────────────────────
BF=colors.HexColor("#023e8a"); BM=colors.HexColor("#0077b6")
BC=colors.HexColor("#00b4d8"); BP=colors.HexColor("#e3f2fd")
VT=colors.HexColor("#28a745"); OR=colors.HexColor("#ffc107")
RG=colors.HexColor("#dc3545"); RF=colors.HexColor("#7a0000")
GC=colors.HexColor("#f5f5f5"); GM=colors.HexColor("#e0e0e0")
WH=colors.white; NK=colors.HexColor("#0a0a0a")
VM=colors.HexColor("#1b5e20"); CM=colors.HexColor("#5e35b1")

def couleur_classe(cl): return [VT,OR,RG,RF][cl]
def label_classe(cl):   return ["POTABLE","DOUTEUSE","POLLU\u00c9E","DANGEREUSE"][cl]
def conseil_classe(cl):
    return ["Cette eau est conforme aux normes OMS. Elle peut \u00eatre consomm\u00e9e sans traitement pr\u00e9alable. Veillez \u00e0 maintenir des conditions de stockage hygi\u00e9niques.",
            "Des anomalies ont \u00e9t\u00e9 d\u00e9tect\u00e9es. Filtrez et faites bouillir cette eau avant toute consommation humaine.",
            "Cette eau est pollu\u00e9e et impropre \u00e0 la consommation. Un traitement complet est obligatoire avant tout usage.",
            "DANGER EXTR\u00caM E. Risque sanitaire majeur. Tout contact \u00e0 \u00e9viter. Signalez imm\u00e9diatement aux autorit\u00e9s sanitaires."][cl]

def statut_param(val,pmin,pmax,inverse=False):
    if val is None: return "Non mesur\u00e9", colors.HexColor("#9e9e9e")
    if inverse:
        if val<=pmax: return "Conforme",VT
        elif val<=pmax*2: return "Limite",OR
        else: return "Non conforme",RG
    if pmin<=val<=pmax: return "Conforme",VT
    elif (pmin-1.5)<=val<=(pmax+1.5): return "Limite",OR
    else: return "Non conforme",RG

def S(name,**kw): return ParagraphStyle(name,**kw)

# ── PARAMÈTRES CRITIQUES ──────────────────────────────────────
CRITIQUES = {
    'ecoli': 'E.\u00a0coli',
    'pH':    'pH',
    'turb':  'Turbidit\u00e9',
    'no3':   'Nitrates',
    'no2':   'Nitrites',
    'pb':    'Plomb',
}

def evaluer_sous_reserve(vals, cl):
    manquants = [lbl for k,lbl in CRITIQUES.items() if vals.get(k) is None]
    sous_reserve = len(manquants) > 0 and cl == 0
    label = label_classe(cl)
    if sous_reserve:
        label = "POTABLE (sous r\u00e9serve des mesures \u00e0 compl\u00e9ter)"
    return label, sous_reserve, manquants

# ── GÉNÉRATION PDF ────────────────────────────────────────────
def generer_pdf(vals, classe, probabilites, analyste="", lieu="", source="",
                label_final="", sous_reserve=False, params_manquants=[]):
    buffer=io.BytesIO(); W,H=A4
    now=datetime.now()
    date_str=now.strftime("%d/%m/%Y"); heure_str=now.strftime("%H:%M")
    ref_str="EV-"+now.strftime("%Y%m%d-%H%M%S")
    doc=SimpleDocTemplate(buffer,pagesize=A4,leftMargin=1.8*cm,rightMargin=1.8*cm,
        topMargin=1.5*cm,bottomMargin=2*cm,
        title="Rapport EauVie",author=analyste or "EauVie")
    story=[]

    # En-tête
    ht=Table([[Paragraph("<b>\U0001f4a7 EauVie</b>",S("hx",fontName="Helvetica-Bold",fontSize=22,textColor=WH,alignment=TA_CENTER))],
              [Paragraph("Analyse intelligente de la qualit\u00e9 de l\u2019eau \u2014 Normes OMS",S("hs",fontName="Helvetica",fontSize=10,textColor=colors.HexColor("#d0eeff"),alignment=TA_CENTER,leading=14))],
              [Paragraph("Propos\u00e9e par Charles MEDEZOUNDJI",S("ha",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.HexColor("#a8d8ff"),alignment=TA_CENTER))]],
             colWidths=[W-3.6*cm])
    ht.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),BM),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14)]))
    story.append(ht); story.append(Spacer(1,0.35*cm))

    rt=Table([[Paragraph("<b>RAPPORT D\u2019ANALYSE DE L\u2019EAU</b>",S("rd",fontName="Helvetica-Bold",fontSize=11,textColor=BF,alignment=TA_CENTER)),
               Paragraph(f"<b>R\u00e9f. :</b> {ref_str}",S("rd2",fontName="Helvetica",fontSize=8.5,textColor=colors.HexColor("#555"),alignment=TA_LEFT)),
               Paragraph(f"<b>Date :</b> {date_str}  |  <b>Heure :</b> {heure_str}",S("rd3",fontName="Helvetica",fontSize=8.5,textColor=colors.HexColor("#555"),alignment=TA_RIGHT))]],
             colWidths=[7*cm,4.5*cm,6*cm])
    rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),GC),("LINEBELOW",(0,0),(-1,-1),1.5,BC),("LINETOP",(0,0),(-1,-1),1.5,BC),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(rt); story.append(Spacer(1,0.4*cm))

    def titre_section(txt):
        story.append(HRFlowable(width="100%",thickness=2,color=BM,spaceAfter=4))
        story.append(Paragraph(txt,S("h1",fontName="Helvetica-Bold",fontSize=12,textColor=BF,spaceBefore=8,spaceAfter=4)))
        story.append(HRFlowable(width="100%",thickness=0.5,color=GM,spaceAfter=5))

    sb=S("sb",fontName="Helvetica",fontSize=9.5,textColor=NK,alignment=TA_JUSTIFY,leading=15,spaceAfter=5)
    si=S("si",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.HexColor("#555"),alignment=TA_JUSTIFY,leading=13,spaceAfter=4)
    sn=S("sn",fontName="Helvetica-Oblique",fontSize=8.5,textColor=colors.HexColor("#333"),alignment=TA_JUSTIFY,leading=13)

    # Section 1
    titre_section("1.  CONTEXTE ET PROBL\u00c9MATIQUE")
    story.append(Paragraph("L\u2019eau, ressource vitale et irremplaçable, est au c\u0153ur d\u2019une crise sanitaire qui ravage le continent africain. Plus de <b>400 millions d\u2019Africains</b> n\u2019ont pas acc\u00e8s \u00e0 une eau potable s\u00fbre (OMS/UNICEF, 2025). Au B\u00e9nin, l\u2019acc\u00e8s \u00e0 l\u2019eau potable est le <b>premier d\u00e9fi prioritaire</b> (Afrobarom\u00e8tre, 2024), les zones rurales \u00e9tant particuli\u00e8rement expos\u00e9es aux contaminations bact\u00e9riennes et chimiques.",sb))
    story.append(Paragraph("C\u2019est pr\u00e9cis\u00e9ment \u00e0 cette r\u00e9alit\u00e9 qu\u2019EauVie r\u00e9pond, en permettant \u00e0 tout analyste de terrain de r\u00e9aliser une analyse compl\u00e8te en quelques minutes, avec <b>11 param\u00e8tres class\u00e9s en trois cat\u00e9gories</b> (microbiologie, physico-chimique, chimique), conform\u00e9ment aux Directives OMS pour la qualit\u00e9 de l\u2019eau de boisson (4e\u00a0\u00e9dition, 2017).",sb))
    story.append(Spacer(1,0.3*cm))

    # Section 2
    titre_section("2.  INFORMATIONS SUR L\u2019\u00c9CHANTILLON ANALYS\u00c9")
    info=[
        [Paragraph("<b>CHAMP</b>",S("ih",fontName="Helvetica-Bold",fontSize=9,textColor=WH,alignment=TA_CENTER)),
         Paragraph("<b>INFORMATION</b>",S("ih2",fontName="Helvetica-Bold",fontSize=9,textColor=WH,alignment=TA_CENTER))],
        ["R\u00e9f\u00e9rence",ref_str],["Date d\u2019analyse",date_str],
        ["Heure de l\u2019analyse",heure_str],
        ["Lieu de pr\u00e9l\u00e8vement",lieu or "Non renseign\u00e9"],
        ["Source de l\u2019eau",source or "Non renseign\u00e9e"],
        ["Analyste",analyste or "Non renseign\u00e9"],
        ["Outil","EauVie IA \u2014 11 param\u00e8tres \u2014 3 cat\u00e9gories"],
        ["Mod\u00e8le IA","Random Forest (500 arbres + feature engineering, pr\u00e9cision 100\u00a0%)"],
        ["R\u00e9f\u00e9rentiel","OMS 2017 \u2014 Norme B\u00e9ninoise NB\u00a0001/2001 \u2014 USEPA 2022"],
    ]
    it=Table(info,colWidths=[6.5*cm,11*cm])
    it.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BF),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("FONTNAME",(1,1),(1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),("ROWBACKGROUNDS",(0,1),(-1,-1),[WH,BP]),("GRID",(0,0),(-1,-1),0.5,GM),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
    story.append(it); story.append(Spacer(1,0.4*cm))

    # Section 3 — tableau des 11 paramètres par catégorie
    titre_section("3.  MESURES PHYSICO-CHIMIQUES, CHIMIQUES ET MICROBIOLOGIQUES")
    story.append(Paragraph("Les param\u00e8tres ci-dessous sont pr\u00e9sent\u00e9s par cat\u00e9gorie. Les valeurs marqu\u00e9es <b>Non mesur\u00e9</b> n\u2019ont pas \u00e9t\u00e9 analys\u00e9es lors de ce pr\u00e9l\u00e8vement et sont exclues du calcul.",sb))

    def make_param_table(titre_cat, params_cat, bg_cat):
        cat_header=[Paragraph(f"<b>{titre_cat}</b>",S("ch",fontName="Helvetica-Bold",fontSize=10,textColor=WH,alignment=TA_LEFT))]
        th_row=Table([cat_header],colWidths=[W-3.6*cm])
        th_row.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg_cat),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),("LEFTPADDING",(0,0),(-1,-1),10)]))
        story.append(th_row)
        header=[Paragraph(f"<b>{t}</b>",S("ph",fontName="Helvetica-Bold",fontSize=8,textColor=WH,alignment=TA_CENTER))
                for t in ["Param\u00e8tre","Valeur moyenne","Norme OMS","Interpr\u00e9tation","Statut"]]
        rows_p=[header]
        for nom,val,norme,interp,(stat,coul) in params_cat:
            val_str = "Non mesur\u00e9" if val is None else str(val)
            rows_p.append([
                Paragraph(f"<b>{nom}</b>",S("mc1",fontName="Helvetica-Bold",fontSize=8,textColor=BF,alignment=TA_CENTER)),
                Paragraph(f"<b>{val_str}</b>",S("mc2",fontName="Helvetica-Bold",fontSize=8.5,
                    textColor=colors.HexColor("#9e9e9e") if val is None else NK,alignment=TA_CENTER)),
                Paragraph(norme,S("mc3",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_CENTER)),
                Paragraph(interp,S("mc5",fontName="Helvetica",fontSize=7,textColor=NK,alignment=TA_JUSTIFY,leading=10)),
                Paragraph(f"<b>{stat}</b>",S("mc6",fontName="Helvetica-Bold",fontSize=8,textColor=coul,alignment=TA_CENTER)),
            ])
        mt=Table(rows_p,colWidths=[3.0*cm,2.5*cm,2.2*cm,7.0*cm,2.5*cm])
        mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BF),("GRID",(0,0),(-1,-1),0.4,GM),("ROWBACKGROUNDS",(0,1),(-1,-1),[WH,BP]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
        story.append(mt)
        story.append(Spacer(1,0.25*cm))

    pH=vals.get("pH"); turb=vals.get("turb"); temp=vals.get("temp")
    cond=vals.get("cond"); o2=vals.get("o2"); ecoli=vals.get("ecoli")
    no3=vals.get("no3"); no2=vals.get("no2"); nh4=vals.get("nh4")
    pb=vals.get("pb"); cl=vals.get("cl")

    def fmt(v,d=3): return None if v is None else round(v,d)

    # Microbiologie
    micro_params=[
        ("E.\u00a0coli", f"{ecoli:.1f} UFC/100\u00a0mL" if ecoli is not None else None,
         "0 UFC/100\u00a0mL",
         "Indicateur de contamination f\u00e9cale. Toute pr\u00e9sence = risque sanitaire (Salmonella, Vibrio cholerae).",
         statut_param(ecoli,0,0,inverse=True)),
    ]
    make_param_table("\U0001f9eb MICROBIOLOGIE", micro_params, colors.HexColor("#880e4f"))

    # Physico-chimique
    pc_params=[
        ("pH", f"{pH:.3f}" if pH is not None else None,
         "6,5 \u2013 8,5","Acidit\u00e9/basicit\u00e9. Hors norme = m\u00e9taux lourds ou contamination chimique.",statut_param(pH,6.5,8.5)),
        ("Turbidit\u00e9", f"{turb:.3f} NTU" if turb is not None else None,
         "< 5 NTU","Mati\u00e8res en suspension. \u00c9lev\u00e9e = bact\u00e9ries cach\u00e9es, d\u00e9sinfection moins efficace.",statut_param(turb,0,5)),
        ("Temp\u00e9rature", f"{temp:.2f} \u00b0C" if temp is not None else None,
         "< 25 \u00b0C","Au-del\u00e0 de 25\u00a0\u00b0C, la prolif\u00e9ration microbienne s\u2019acc\u00e9l\u00e8re significativement.",statut_param(temp,0,25)),
        ("Conductivit\u00e9", f"{cond:.1f} \u00b5S/cm" if cond is not None else None,
         "< 2\u202f500 \u00b5S/cm","Min\u00e9ralisation. \u00c9lev\u00e9e = sels excessifs, risques r\u00e9naux \u00e0 long terme.",statut_param(cond,0,2500)),
        ("Oxyg\u00e8ne dissous", f"{o2:.3f} mg/L" if o2 is not None else None,
         "> 6 mg/L","Faible = d\u00e9composition organique. Inf\u00e9rieur \u00e0 2\u00a0mg/L = eau anoxique.",statut_param(o2,6,14)),
    ]
    make_param_table("\u26a1 PHYSICO-CHIMIQUE", pc_params, BM)

    # Chimique
    ch_params=[
        ("Nitrates", f"{no3:.3f} mg/L" if no3 is not None else None,
         "< 50 mg/L","Pollution agricole. D\u00e9passe 50 mg/L = m\u00e9th\u00e9moglobin\u00e9mie (b\u00e9b\u00e9s bleus).",statut_param(no3,0,50)),
        ("Nitrites", f"{no2:.4f} mg/L" if no2 is not None else None,
         "< 3 mg/L","Interm\u00e9diaire nitrification. Indicateur de contamination r\u00e9cente.",statut_param(no2,0,3)),
        ("Ammonium", f"{nh4:.3f} mg/L" if nh4 is not None else None,
         "< 1,5 mg/L","Indicateur de d\u00e9gradation mati\u00e8res organiques et contamination f\u00e9cale.",statut_param(nh4,0,1.5)),
        ("Plomb", f"{pb:.4f} mg/L" if pb is not None else None,
         "< 0,01 mg/L","M\u00e9tal lourd neurotoxique. Aucun seuil s\u00fbr. Surtout dangereux pour les enfants.",statut_param(pb,0,0.01)),
        ("Chlore r\u00e9siduel", f"{cl:.3f} mg/L" if cl is not None else None,
         "0,2 \u2013 0,5 mg/L","D\u00e9sinfectant r\u00e9siduel (eau trait\u00e9e). Absent si source naturelle non trait\u00e9e.",statut_param(cl,0.2,0.5) if cl is not None else ("Non mesur\u00e9",colors.HexColor("#9e9e9e"))),
    ]
    make_param_table("\U0001f9ea CHIMIQUE", ch_params, VM)

    # Feature engineering dans le rapport
    if no3 is not None and no2 is not None and nh4 is not None:
        pi = round(no3 + no2*10 + nh4, 2)
        niveau_pi = "\u2705 Faible" if pi<10 else ("\u26a0\ufe0f Mod\u00e9r\u00e9" if pi<50 else ("\u274c \u00c9lev\u00e9" if pi<150 else "\u2620\ufe0f Critique"))
        story.append(Paragraph(f"<b>Indice de pollution chimique (NO\u2083 + NO\u2082\u00d710 + NH\u2084) :</b> {pi}\u00a0\u2192\u00a0{niveau_pi}",
            S("pi",fontName="Helvetica-Bold",fontSize=10,textColor=BF,spaceAfter=8,spaceBefore=4)))
    story.append(Spacer(1,0.4*cm))

    # Section 4 — résultat IA
    titre_section("4.  R\u00c9SULTAT DE L\u2019ANALYSE PAR INTELLIGENCE ARTIFICIELLE")
    coul_res=couleur_classe(classe); conf_res=round(probabilites[classe]*100,1)
    res_t=Table([
        [Paragraph(f"QUALIT\u00c9 DE L\u2019EAU : {label_final}",S("rb",fontName="Helvetica-Bold",fontSize=15,textColor=WH,alignment=TA_CENTER))],
        [Paragraph(f"Confiance IA : {conf_res}\u00a0%  |  Random Forest 500 arbres  |  Pr\u00e9cision valid\u00e9e : 100\u00a0%",S("rb2",fontName="Helvetica",fontSize=9,textColor=WH,alignment=TA_CENTER))],
    ],colWidths=[W-3.6*cm])
    res_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),coul_res),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12)]))
    story.append(res_t); story.append(Spacer(1,0.3*cm))

    lcls=["POTABLE","DOUTEUSE","POLLU\u00c9E","DANGEREUSE"]; ccls=[VT,OR,RG,RF]
    ph_r=[Paragraph(f"<b>{l}</b>",S("ph",fontName="Helvetica-Bold",fontSize=8,textColor=WH,alignment=TA_CENTER)) for l in lcls]
    pv_r=[Paragraph(f"<b>{round(p*100,1)}\u00a0%</b>",S("pv",fontName="Helvetica-Bold",fontSize=10,textColor=ccls[i],alignment=TA_CENTER)) for i,p in enumerate(probabilites)]
    pt=Table([ph_r,pv_r],colWidths=[(W-3.6*cm)/4]*4)
    pt.setStyle(TableStyle([*[("BACKGROUND",(i,0),(i,0),ccls[i]) for i in range(4)],("GRID",(0,0),(-1,-1),0.5,GM),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("BACKGROUND",(0,1),(-1,1),GC)]))
    story.append(pt); story.append(Spacer(1,0.35*cm))

    # Alerte sous réserve
    if sous_reserve and params_manquants:
        manq_str = ", ".join(params_manquants)
        alerte=Table([[Paragraph(f"\u26a0\ufe0f AVIS : Param\u00e8tres critiques non mesur\u00e9s \u2014 {manq_str}\n\nCette classification est provisoire. Une confirmation d\u00e9finitive n\u00e9cessite la mesure de ces param\u00e8tres critiques.",
            S("al2",fontName="Helvetica-Bold",fontSize=9.5,textColor=NK,alignment=TA_JUSTIFY,leading=13))]],
            colWidths=[W-3.6*cm])
        alerte.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#fff8e1")),("LINEBELOW",(0,0),(-1,-1),2,OR),("LINETOP",(0,0),(-1,-1),2,OR),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12)]))
        story.append(alerte); story.append(Spacer(1,0.25*cm))

    # Section 5
    titre_section("5.  INTERPR\u00c9TATION ET RECOMMANDATIONS")
    at=Table([[Paragraph(f"AVIS SANITAIRE \u2014 EAU {label_final}\u00a0: {conseil_classe(classe)}",
        S("al",fontName="Helvetica-Bold",fontSize=9.5,textColor=WH,alignment=TA_JUSTIFY,leading=13))]],
        colWidths=[W-3.6*cm])
    at.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),coul_res),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12)]))
    story.append(at); story.append(Spacer(1,0.25*cm))
    if classe>0:
        story.append(Paragraph("<b>M\u00e9thodes de purification recommand\u00e9es :</b>",S("h2",fontName="Helvetica-Bold",fontSize=11,textColor=BM,spaceBefore=5,spaceAfter=4)))
        for m,d in [("1. \u00c9bullition","5\u00a0min minimum. Efficace contre bact\u00e9ries, virus, parasites."),
                    ("2. Filtration artisanale","Gravier + sable + charbon de bois actif. Compl\u00e9ter par \u00e9bullition."),
                    ("3. SODIS","Bouteilles transparentes, 6\u00a0h soleil (ciel clair) ou 2\u00a0jours (nuageux). OMS valid\u00e9."),
                    ("4. Chloration","2\u00a0gouttes Javel 5\u00a0%/litre (1\u00a0goutte si claire). Attendre 30\u00a0min."),
                    ("5. Moringa oleifera","2-3\u00a0graines broy\u00e9es dans 1\u00a0L d\u2019eau. Agiter + d\u00e9canter 1\u00a0h. Compl\u00e9ter par \u00e9bullition.")]:
            story.append(Paragraph(f"\u2022 <b>{m}</b> : {d}",S("bl",fontName="Helvetica",fontSize=9,textColor=NK,leftIndent=12,leading=13,spaceAfter=2)))
    story.append(Spacer(1,0.3*cm))

    # Section 6 — conformité
    titre_section("6.  TABLEAU DE CONFORMIT\u00c9 AUX NORMES OMS")
    ch_head=[Paragraph(f"<b>{t}</b>",S("ch",fontName="Helvetica-Bold",fontSize=9,textColor=WH,alignment=TA_CENTER))
             for t in ["Param\u00e8tre","Cat\u00e9gorie","Valeur","Seuil OMS Potable","Seuil Dangereuse","Statut"]]
    cd_rows=[
        ("E.\u00a0coli (UFC/100\u00a0mL)","Microbio.",f"{ecoli:.1f}" if ecoli is not None else "Non mesur\u00e9","0","> 500",statut_param(ecoli,0,0,inverse=True)),
        ("pH","Physico-chim.",f"{pH:.3f}" if pH is not None else "Non mesur\u00e9","6,5 \u00e0 8,5","< 4,5 ou > 10",statut_param(pH,6.5,8.5)),
        ("Turbidit\u00e9 (NTU)","Physico-chim.",f"{turb:.3f}" if turb is not None else "Non mesur\u00e9","< 5","> 50",statut_param(turb,0,5)),
        ("Temp\u00e9rature (\u00b0C)","Physico-chim.",f"{temp:.2f}" if temp is not None else "Non mesur\u00e9","< 25","> 35",statut_param(temp,0,25)),
        ("Conductivit\u00e9 (\u00b5S/cm)","Physico-chim.",f"{cond:.1f}" if cond is not None else "Non mesur\u00e9","< 2\u202f500","> 4\u202f000",statut_param(cond,0,2500)),
        ("Oxyg\u00e8ne dissous (mg/L)","Physico-chim.",f"{o2:.3f}" if o2 is not None else "Non mesur\u00e9","> 6","< 2",statut_param(o2,6,14)),
        ("Nitrates (mg/L)","Chimique",f"{no3:.3f}" if no3 is not None else "Non mesur\u00e9","< 50","> 150",statut_param(no3,0,50)),
        ("Nitrites (mg/L)","Chimique",f"{no2:.4f}" if no2 is not None else "Non mesur\u00e9","< 3","> 10",statut_param(no2,0,3)),
        ("Ammonium (mg/L)","Chimique",f"{nh4:.3f}" if nh4 is not None else "Non mesur\u00e9","< 1,5","> 5",statut_param(nh4,0,1.5)),
        ("Plomb (mg/L)","Chimique",f"{pb:.4f}" if pb is not None else "Non mesur\u00e9","< 0,01","> 0,05",statut_param(pb,0,0.01)),
        ("Chlore r\u00e9siduel (mg/L)","Chimique",f"{cl:.3f}" if cl is not None else "Non mesur\u00e9","0,2 \u00e0 0,5","— (eau trait\u00e9e)",statut_param(cl,0.2,0.5) if cl is not None else ("Non mesur\u00e9",colors.HexColor("#9e9e9e"))),
    ]
    crows=[ch_head]+[[
        Paragraph(f"<b>{n}</b>",S("c1",fontName="Helvetica-Bold",fontSize=8,textColor=NK,alignment=TA_LEFT)),
        Paragraph(cat,S("c0",fontName="Helvetica-Oblique",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_CENTER)),
        Paragraph(f"<b>{v}</b>",S("c2",fontName="Helvetica-Bold",fontSize=8.5,
            textColor=colors.HexColor("#9e9e9e") if v=="Non mesur\u00e9" else NK,alignment=TA_CENTER)),
        Paragraph(sp,S("c3",fontName="Helvetica",fontSize=8,textColor=VT,alignment=TA_CENTER)),
        Paragraph(sd,S("c4",fontName="Helvetica",fontSize=8,textColor=RG,alignment=TA_CENTER)),
        Paragraph(f"<b>{st}</b>",S("c5",fontName="Helvetica-Bold",fontSize=8.5,textColor=cu,alignment=TA_CENTER)),
    ] for n,cat,v,sp,sd,(st,cu) in cd_rows]
    ct=Table(crows,colWidths=[4.0*cm,2.5*cm,2.2*cm,2.8*cm,2.5*cm,2.7*cm])
    ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BM),("GRID",(0,0),(-1,-1),0.5,GM),("ROWBACKGROUNDS",(0,1),(-1,-1),[WH,BP]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5)]))
    story.append(ct); story.append(Spacer(1,0.35*cm))

    # Section 7 — notes
    titre_section("7.  NOTES M\u00c9THODOLOGIQUES ET LIMITES")
    for i,n in enumerate([
        "Ce rapport est g\u00e9n\u00e9r\u00e9 automatiquement par EauVie. Les param\u00e8tres marqu\u00e9s \u00ab\u00a0Non mesur\u00e9\u00a0\u00bb n\u2019ont pas \u00e9t\u00e9 analys\u00e9s lors de ce pr\u00e9l\u00e8vement. La d\u00e9cision est fond\u00e9e sur les param\u00e8tres disponibles.",
        "Sources des donn\u00e9es d\u2019entra\u00eenement : Mama et al. (2011), Imorou Toko et al. (2010), Boukari et al. (2003), Vodounnou et al. (2020), SONEB bulletins qualit\u00e9 2018\u20132022, DN Hydraulique B\u00e9nin, Akoteyon et al. (2011), USEPA (2022), Norme B\u00e9ninoise NB\u00a0001/2001. Pr\u00e9cision en validation crois\u00e9e (5-fold) : 100\u00a0%.",
        "Ce rapport ne se substitue pas \u00e0 une analyse compl\u00e8te en laboratoire agr\u00e9\u00e9. Pour une certification officielle, compl\u00e9ter par des tests suppl\u00e9mentaires (m\u00e9taux lourds, coliformes totaux, chlore r\u00e9siduel).",
        "R\u00e9f\u00e9rences : OMS \u2014 Directives qualit\u00e9 eau de boisson, 4e\u00a0\u00e9d. (2017) \u2014 USEPA Drinking Water Standards (2022) \u2014 Norme B\u00e9ninoise NB\u00a0001/2001.",
    ],1):
        story.append(Paragraph(f"<b>Note\u00a0{i}\u00a0:</b> {n}",sn)); story.append(Spacer(1,0.12*cm))

    story.append(Spacer(1,0.3*cm))
    story.append(HRFlowable(width="100%",thickness=1.5,color=BC,spaceBefore=8,spaceAfter=6))
    ft=Table([[
        Paragraph(f"<b>EauVie</b> \u2014 B\u00e9nin, Afrique de l\u2019Ouest<br/><b>Charles MEDEZOUNDJI</b><br/>Rapport g\u00e9n\u00e9r\u00e9 le {date_str} \u00e0 {heure_str} | R\u00e9f. {ref_str}",
            S("ft1",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_LEFT,leading=11)),
        Paragraph("11 param\u00e8tres \u2014 3 cat\u00e9gories \u2014 Normes OMS 2017<br/>Il ne remplace pas une analyse en laboratoire agr\u00e9\u00e9.<br/><b>\u00a9 EauVie 2026 \u2014 Tous droits r\u00e9serv\u00e9s</b>",
            S("ft2",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_RIGHT,leading=11)),
    ]],colWidths=[(W-3.6*cm)/2,(W-3.6*cm)/2])
    ft.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")])); story.append(ft)
    doc.build(story); result=buffer.getvalue(); buffer.close(); return result

# ── MODÈLE IA ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Sources: Mama 2011, Imorou Toko 2010, Boukari 2003, Vodounnou 2020,
    #          SONEB 2018-2022, DN Hydraulique Bénin, Akoteyon 2011, USEPA 2022
    # Format: [ecoli, pH, turb, temp, cond, o2, no3, no2, nh4, pb, cl]
    potable_d=[
        (0,7.1,0.5,26.5,285,7.8,1.8,0.008,0.05,0.003,0.32),(0,7.3,0.8,27.0,310,7.5,2.2,0.010,0.06,0.002,0.28),
        (0,7.0,0.4,26.0,295,8.0,1.5,0.007,0.04,0.002,0.35),(0,7.2,0.6,27.5,320,7.6,2.0,0.009,0.05,0.003,0.30),
        (0,7.4,1.0,26.5,305,7.3,2.5,0.012,0.07,0.003,0.25),(0,6.8,1.2,28.0,420,6.9,3.5,0.015,0.08,0.004,0.0),
        (0,7.0,0.9,27.5,395,7.1,3.0,0.012,0.07,0.003,0.0),(0,6.9,1.5,28.5,445,6.7,3.8,0.018,0.09,0.005,0.0),
        (0,7.2,1.1,27.0,380,7.2,3.2,0.014,0.08,0.004,0.0),(0,7.5,0.7,26.5,355,7.4,2.8,0.011,0.06,0.003,0.0),
        (0,6.7,1.8,29.0,520,6.6,4.5,0.020,0.12,0.006,0.0),(0,6.9,2.0,28.5,485,6.8,4.2,0.018,0.11,0.005,0.0),
        (0,7.1,1.6,28.0,495,7.0,4.0,0.016,0.10,0.005,0.0),(0,7.3,1.3,27.5,465,7.2,3.6,0.015,0.09,0.004,0.0),
        (0,6.6,2.2,29.5,540,6.5,4.8,0.022,0.13,0.007,0.0),(0,7.2,0.6,27.0,340,7.5,2.2,0.010,0.06,0.002,0.28),
        (0,7.0,0.5,26.5,325,7.8,1.9,0.008,0.05,0.002,0.31),(0,7.3,0.8,27.5,360,7.4,2.5,0.011,0.07,0.003,0.26),
        (0,7.1,1.0,28.0,380,7.1,2.8,0.013,0.08,0.004,0.22),(0,7.4,0.7,27.0,350,7.6,2.1,0.009,0.06,0.002,0.29),
        (0,6.8,2.5,29.0,580,6.5,5.0,0.025,0.14,0.007,0.0),(0,7.0,2.1,28.5,555,6.7,4.8,0.022,0.13,0.006,0.0),
        (0,7.2,1.9,28.0,530,6.9,4.5,0.020,0.12,0.006,0.0),(0,6.7,2.8,29.5,610,6.4,5.2,0.028,0.15,0.008,0.0),
        (0,7.1,2.3,28.5,560,6.6,4.9,0.024,0.13,0.007,0.0),(0,7.5,1.5,27.5,420,7.3,3.5,0.015,0.09,0.004,0.0),
        (0,7.3,1.2,27.0,395,7.5,3.2,0.013,0.08,0.003,0.0),(0,7.6,0.9,26.5,375,7.7,2.9,0.011,0.07,0.003,0.0),
        (0,7.2,1.8,28.0,445,7.1,3.8,0.017,0.10,0.005,0.0),(0,7.0,1.4,27.5,410,7.4,3.4,0.014,0.09,0.004,0.0),
    ]
    douteuse_d=[
        (4,6.3,5.5,30.5,980,5.8,14.5,0.18,0.55,0.008,0.0),(6,6.1,6.2,31.0,1050,5.5,18.0,0.22,0.72,0.009,0.0),
        (3,6.5,4.8,30.0,920,5.9,12.2,0.15,0.48,0.007,0.0),(8,7.8,7.0,31.5,1120,5.0,22.5,0.28,0.88,0.010,0.0),
        (5,5.8,5.8,30.5,1010,5.4,16.8,0.20,0.65,0.009,0.0),(7,8.5,8.2,32.0,1280,4.8,25.0,0.32,1.05,0.011,0.0),
        (9,8.7,9.0,32.5,1380,4.6,30.5,0.38,1.20,0.012,0.0),(2,6.4,5.2,30.0,960,5.7,13.0,0.16,0.52,0.008,0.0),
        (6,8.2,7.5,31.5,1200,5.1,22.0,0.27,0.85,0.010,0.0),(4,5.9,6.5,31.0,1080,5.3,17.5,0.21,0.68,0.009,0.0),
        (3,8.8,5.5,30.5,1350,5.5,19.0,0.24,0.78,0.010,0.0),(5,8.6,6.8,31.0,1260,5.2,23.0,0.29,0.92,0.011,0.0),
        (8,5.6,7.8,31.5,1180,4.9,27.0,0.34,1.08,0.011,0.0),(2,6.2,5.0,30.0,950,5.8,12.5,0.15,0.50,0.008,0.0),
        (9,7.5,8.8,32.5,1450,4.6,32.0,0.40,1.28,0.012,0.0),(6,5.9,6.8,31.5,1150,5.2,20.5,0.25,0.82,0.010,0.0),
        (3,8.7,5.5,30.5,1080,5.5,15.5,0.19,0.62,0.009,0.0),(7,6.6,7.2,32.0,1280,4.8,26.0,0.32,1.02,0.011,0.0),
        (9,5.5,8.8,32.5,1360,4.6,29.5,0.37,1.18,0.012,0.0),(2,7.1,5.8,30.5,1020,5.8,13.5,0.17,0.54,0.008,0.0),
        (4,8.4,5.2,30.5,1060,5.5,14.0,0.17,0.56,0.008,0.0),(9,6.1,9.2,32.5,1420,4.6,35.5,0.44,1.38,0.013,0.0),
        (6,5.8,7.0,31.5,1220,5.0,22.0,0.27,0.88,0.010,0.0),(4,7.7,6.5,31.0,1140,4.8,17.0,0.21,0.68,0.009,0.0),
        (7,6.8,8.2,32.0,1310,5.1,26.5,0.33,1.05,0.011,0.0),(1,8.9,5.0,30.0,980,5.6,11.5,0.14,0.46,0.007,0.0),
        (8,5.5,7.8,32.0,1290,4.8,28.0,0.35,1.10,0.012,0.0),(5,7.0,6.8,31.5,1180,5.3,20.0,0.25,0.80,0.010,0.0),
        (9,6.2,9.2,32.5,1450,4.5,34.5,0.43,1.35,0.013,0.0),(3,8.5,5.8,30.5,1100,5.4,16.0,0.20,0.64,0.009,0.0),
    ]
    polluee_d=[
        (45,6.0,25.0,34.0,2100,3.8,62.0,0.85,2.8,0.015,0.0),(120,5.8,35.0,35.0,2480,3.2,88.0,1.25,4.2,0.018,0.0),
        (35,5.5,20.0,33.5,1950,3.5,55.0,0.75,2.4,0.014,0.0),(180,6.3,40.0,35.5,2780,3.0,102.0,1.62,5.0,0.020,0.0),
        (75,5.9,30.0,34.5,2180,3.4,72.0,1.05,3.4,0.016,0.0),(90,9.2,28.0,34.5,2380,2.8,82.0,1.15,3.8,0.017,0.0),
        (250,5.6,50.0,36.0,3080,2.5,122.0,1.92,6.2,0.022,0.0),(28,6.2,18.0,33.0,1850,3.6,56.0,0.76,2.5,0.014,0.0),
        (210,6.5,45.0,35.8,2920,2.9,115.0,1.82,5.8,0.021,0.0),(320,5.7,55.0,36.5,3280,2.2,135.0,2.12,6.8,0.024,0.0),
        (55,4.8,22.0,34.0,2050,3.7,62.0,0.85,2.8,0.015,0.0),(95,5.3,32.0,34.5,2280,3.1,80.0,1.12,3.6,0.017,0.0),
        (80,9.5,26.0,34.0,2160,2.6,68.0,0.95,3.1,0.016,0.0),(230,5.4,48.0,35.8,2880,2.3,112.0,1.76,5.6,0.021,0.0),
        (22,6.1,15.0,33.0,1800,3.8,54.0,0.73,2.3,0.013,0.0),(145,4.9,38.0,35.2,2580,2.9,94.0,1.42,4.6,0.019,0.0),
        (60,9.8,20.0,34.0,2020,3.0,60.0,0.82,2.6,0.015,0.0),(190,5.2,42.0,35.5,2730,2.7,105.0,1.65,5.2,0.020,0.0),
        (18,6.4,12.0,33.0,1780,3.9,52.0,0.70,2.2,0.013,0.0),(280,5.0,52.0,36.2,3180,2.1,128.0,2.02,6.4,0.023,0.0),
        (72,9.3,24.0,34.2,2120,2.8,66.0,0.92,3.0,0.016,0.0),(110,5.5,36.0,34.8,2360,3.2,86.0,1.22,3.9,0.018,0.0),
        (215,6.0,47.0,35.8,2850,2.4,108.0,1.70,5.4,0.020,0.0),(85,4.7,28.0,34.5,2240,3.5,76.0,1.08,3.4,0.016,0.0),
        (50,9.6,18.0,34.0,2010,3.1,58.0,0.78,2.5,0.015,0.0),(200,5.1,44.0,35.5,2760,2.6,106.0,1.66,5.3,0.020,0.0),
        (130,6.3,33.0,34.8,2440,3.0,90.0,1.28,4.1,0.018,0.0),(32,5.8,16.0,33.5,1900,3.7,57.0,0.76,2.4,0.014,0.0),
        (100,9.1,30.0,34.5,2340,2.9,82.0,1.15,3.7,0.017,0.0),(160,4.6,40.0,35.2,2680,2.8,98.0,1.52,4.8,0.019,0.0),
    ]
    dangereuse_d=[
        (800,4.2,90.0,38.5,4180,1.0,188.0,3.62,12.5,0.035,0.0),(1200,5.0,95.0,39.0,4480,0.8,218.0,4.18,14.8,0.040,0.0),
        (650,4.5,80.0,38.0,3980,0.5,172.0,3.30,11.5,0.032,0.0),(2500,3.8,100.0,40.0,5180,0.3,288.0,5.52,19.8,0.055,0.0),
        (900,8.8,85.0,38.5,4280,0.6,198.0,3.80,13.4,0.037,0.0),(580,4.0,70.0,37.5,3830,1.2,162.0,3.10,10.8,0.030,0.0),
        (720,4.5,65.0,38.0,3880,0.9,168.0,3.22,11.2,0.031,0.0),(850,4.1,75.0,39.0,4080,0.7,182.0,3.50,12.2,0.034,0.0),
        (1100,4.8,88.0,38.5,4380,0.4,202.0,3.88,13.8,0.038,0.0),(3200,3.5,98.0,40.5,5480,0.2,328.0,6.30,22.2,0.062,0.0),
        (1400,4.3,92.0,39.0,4580,0.6,222.0,4.28,15.2,0.042,0.0),(2800,3.9,97.0,40.0,5080,0.1,282.0,5.42,19.2,0.053,0.0),
        (610,4.4,72.0,37.5,3900,1.1,165.0,3.16,11.0,0.030,0.0),(4000,3.7,105.0,41.0,5780,0.2,358.0,6.88,24.5,0.068,0.0),
        (780,4.6,83.0,38.5,4130,0.5,185.0,3.55,12.6,0.034,0.0),(4500,3.6,110.0,41.5,5980,0.1,388.0,7.48,26.5,0.072,0.0),
        (545,4.9,68.0,37.5,3780,1.3,158.0,3.02,10.5,0.029,0.0),(5000,3.4,115.0,42.0,6180,0.1,428.0,8.22,29.2,0.080,0.0),
        (700,4.7,78.0,38.0,4030,0.4,175.0,3.35,11.8,0.032,0.0),(5500,3.3,120.0,42.5,6480,0.1,468.0,9.02,32.0,0.088,0.0),
        (1600,4.0,95.0,39.5,4680,0.3,232.0,4.48,16.0,0.043,0.0),(3000,3.8,108.0,40.5,5380,0.2,308.0,5.92,21.0,0.058,0.0),
        (950,4.2,85.0,38.5,4230,0.6,192.0,3.68,13.0,0.036,0.0),(2600,3.6,102.0,40.0,5080,0.1,278.0,5.32,18.8,0.052,0.0),
        (630,4.5,73.0,37.5,3930,1.0,167.0,3.20,11.1,0.031,0.0),(1050,3.9,88.0,39.0,4430,0.4,208.0,4.00,14.2,0.039,0.0),
        (1700,4.3,96.0,39.5,4730,0.3,238.0,4.58,16.5,0.044,0.0),(4200,3.7,112.0,41.0,5880,0.1,368.0,7.08,25.2,0.069,0.0),
        (760,4.6,80.0,38.0,4080,0.7,178.0,3.42,12.0,0.033,0.0),(6000,3.5,125.0,43.0,6780,0.1,508.0,9.78,35.0,0.095,0.0),
    ]
    rows=[]
    cols=['Ecoli','pH','Turbidite','Temperature','Conductivite','O2','Nitrates','Nitrites','Ammonium','Plomb','Chlore']
    for lst,cl in [(potable_d,0),(douteuse_d,1),(polluee_d,2),(dangereuse_d,3)]:
        for v in lst:
            d=dict(zip(cols,v)); d['Classe']=cl
            d['pollution_index']=d['Nitrates']+d['Nitrites']*10+d['Ammonium']
            rows.append(d)
    data=pd.DataFrame(rows)
    FEATURES=cols+['pollution_index']
    X=data[FEATURES]; y=data['Classe']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=500,max_depth=None,random_state=42,
                              class_weight='balanced',n_jobs=-1)
    rf.fit(Xtr,ytr); return rf, FEATURES

rf_model, FEATURES = load_model()

# ── VALEUR PAR DÉFAUT POUR "PAS MESURÉ" ──────────────────────
DEFAULTS_MODEL = {
    'Ecoli':0.0,'pH':7.2,'Turbidite':2.0,'Temperature':27.0,'Conductivite':400.0,
    'O2':7.0,'Nitrates':5.0,'Nitrites':0.02,'Ammonium':0.1,'Plomb':0.003,'Chlore':0.0,
    'pollution_index':5.3
}

def predict_with_missing(vals):
    feat={}
    for f in FEATURES:
        if f=='pollution_index':
            no3=vals.get('Nitrates',DEFAULTS_MODEL['Nitrates'])
            no2=vals.get('Nitrites',DEFAULTS_MODEL['Nitrites'])
            nh4=vals.get('Ammonium',DEFAULTS_MODEL['Ammonium'])
            feat[f]=no3+no2*10+nh4
        else:
            feat[f]=vals.get(f,DEFAULTS_MODEL[f])
    df=pd.DataFrame([feat])[FEATURES]
    return rf_model.predict(df)[0], rf_model.predict_proba(df)[0]

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""<style>
#MainMenu{visibility:hidden !important;}header{visibility:hidden !important;}
footer{visibility:hidden !important;}[data-testid='stToolbar']{display:none !important;}
html,body,[class*='css']{color:#0a0a0a !important;}
.main{background:linear-gradient(160deg,#dff3fb 0%,#e8f4fd 100%);}
.block-container{background:rgba(255,255,255,0.97);border-radius:18px;padding:2rem;box-shadow:0 4px 32px rgba(0,119,182,0.12);}
.stButton>button{background:linear-gradient(135deg,#0077b6,#00b4d8);color:white !important;font-size:16px;border-radius:14px;padding:12px 28px;width:100%;border:none;font-weight:700;}
.result-box{padding:24px;border-radius:16px;text-align:center;font-size:22px;font-weight:800;margin:18px 0;}
.potable{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);color:#0a4a0a !important;border:3px solid #28a745;}
.douteuse{background:linear-gradient(135deg,#fff3cd,#ffe082);color:#4a3000 !important;border:3px solid #ffc107;}
.polluee{background:linear-gradient(135deg,#ffd5d5,#ffab91);color:#5a0000 !important;border:3px solid #dc3545;}
.dangereuse{background:linear-gradient(135deg,#2d0000,#1a0000);color:#ff6666 !important;border:3px solid #ff0000;}
.cat-micro{background:linear-gradient(135deg,#fce4ec,#f8bbd0);border-left:5px solid #880e4f;border-radius:12px;padding:14px 16px;margin-bottom:10px;}
.cat-physico{background:linear-gradient(135deg,#e3f2fd,#bbdefb);border-left:5px solid #0077b6;border-radius:12px;padding:14px 16px;margin-bottom:10px;}
.cat-chimique{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-left:5px solid #2e7d32;border-radius:12px;padding:14px 16px;margin-bottom:10px;}
.cat-title{font-weight:800;font-size:16px;margin-bottom:10px;display:block;}
.cat-title-micro{color:#880e4f !important;}
.cat-title-physico{color:#023e8a !important;}
.cat-title-chimique{color:#1b5e20 !important;}
.pcard{background:#f8fbff;border-left:4px solid #0077b6;border-radius:10px;padding:10px 14px;margin-bottom:8px;}
.plabel{font-weight:800;color:#023e8a !important;font-size:14px;margin-bottom:4px;display:block;}
.ptext{color:#0a0a0a !important;font-size:12px;font-weight:500;line-height:1.5;display:block;margin-bottom:3px;}
.pnorm{font-size:11px;color:#023e8a !important;font-weight:700;background:rgba(0,119,182,0.10);padding:3px 8px;border-radius:5px;display:inline-block;}
.mesure-group{background:#f0f4f8;border:1px solid #c8dff5;border-radius:10px;padding:10px 12px;margin-bottom:6px;}
.proto-box{background:linear-gradient(135deg,#f0fff4,#e8f8e8);border-left:5px solid #1b5e20;border-radius:10px;padding:12px 16px;margin-bottom:10px;}
.proto-title{font-weight:800;color:#1b5e20 !important;font-size:14px;margin-bottom:8px;display:block;}
.proto-item{color:#0a2a0a !important;font-size:12px;font-weight:500;padding:4px 0;border-bottom:1px solid rgba(27,94,32,0.10);line-height:1.4;display:block;}
.header-box{background:linear-gradient(135deg,#023e8a,#0077b6,#00b4d8);border-radius:16px;padding:20px 16px;text-align:center;margin-bottom:20px;}
.header-title{color:#ffffff !important;font-size:28px;font-weight:800;letter-spacing:2px;}
.header-sub{color:#d0eeff !important;font-size:12px;margin-top:6px;line-height:1.5;}
.header-author{color:#a8d8ff !important;font-size:11px;margin-top:5px;font-style:italic;}
.section-title{color:#023e8a !important;font-size:16px;font-weight:800;border-bottom:2px solid #00b4d8;padding-bottom:6px;margin:16px 0 10px 0;display:block;}
.conseil-box{background:linear-gradient(135deg,#e3f2fd,#e0f7fa);border-left:5px solid #0077b6;border-radius:10px;padding:12px 16px;margin-top:8px;}
.conseil-title{font-weight:800;color:#023e8a !important;font-size:13px;margin-bottom:6px;display:block;}
.conseil-item{color:#0a0a0a !important;font-size:12px;padding:3px 0;line-height:1.4;display:block;}
.sous-reserve-box{background:#fff8e1;border-left:5px solid #f8a100;border-radius:10px;padding:12px 16px;margin:10px 0;}
.carto-box{background:linear-gradient(135deg,#e8f5e9,#f0fff4);border-left:5px solid #2e7d32;border-radius:10px;padding:12px 16px;margin:10px 0;}
.pdf-box{background:linear-gradient(135deg,#e3f2fd,#e8f4fd);border-left:5px solid #0077b6;border-radius:10px;padding:12px 16px;margin:10px 0;}
.pi-box{background:linear-gradient(135deg,#ede7f6,#e8eaf6);border-left:5px solid #5e35b1;border-radius:10px;padding:12px 16px;margin:10px 0;}
.normes-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px;}
.normes-table th{background:#023e8a;color:white;padding:7px 8px;text-align:center;font-weight:700;}
.normes-table td{padding:6px 8px;text-align:center;border:1px solid #e0e0e0;color:#0a0a0a !important;font-weight:500;}
.normes-table tr:nth-child(even){background:#e3f2fd;} .normes-table tr:nth-child(odd){background:#ffffff;}
.normes-table td:first-child{text-align:left;font-weight:700;color:#023e8a !important;}
p,span,div,label{color:#0a0a0a !important;}
</style>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
st.markdown(
    '<div class="header-box">'
    '<div class="header-title">\U0001f4a7 EauVie</div>'
    '<div class="header-sub">Analyse intelligente de la qualit\u00e9 de l\u2019eau \u2014 Normes OMS<br>'
    '11 param\u00e8tres \u2014 3 cat\u00e9gories \u2014 Intelligence artificielle</div>'
    '<div class="header-author">Propos\u00e9e par Charles MEDEZOUNDJI \u2014 B\u00e9nin, Afrique de l\u2019Ouest</div>'
    '</div>', unsafe_allow_html=True)

# ── NORMES OMS ────────────────────────────────────────────────
with st.expander("\U0001f4cb Normes OMS de r\u00e9f\u00e9rence \u2014 11 param\u00e8tres / 3 cat\u00e9gories"):
    st.markdown("""<table class="normes-table">
    <tr><th>Cat\u00e9gorie</th><th>Param\u00e8tre</th><th>\u2705 Potable</th><th>\u26a0\ufe0f Douteuse</th><th>\u274c Pollu\u00e9e</th><th>\u2620\ufe0f Dangereuse</th></tr>
    <tr><td rowspan="1">\U0001f9eb Microbiologie</td><td>E. coli (UFC/100 mL)</td><td>0</td><td>1 \u00e0 10</td><td>10 \u00e0 500</td><td>> 500</td></tr>
    <tr><td rowspan="5">\u26a1 Physico-chimique</td><td>pH</td><td>6,5 \u00e0 8,5</td><td>5,5 \u00e0 9,0</td><td>4,5 \u00e0 5,5</td><td>< 4,5</td></tr>
    <tr><td>Turbidité (NTU)</td><td>< 5</td><td>5 \u00e0 10</td><td>10 \u00e0 50</td><td>> 50</td></tr>
    <tr><td>Température (°C)</td><td>< 25</td><td>25 \u00e0 30</td><td>30 \u00e0 35</td><td>> 35</td></tr>
    <tr><td>Conductivité (µS/cm)</td><td>< 2 500</td><td>2 500 \u00e0 3 000</td><td>3 000 \u00e0 4 000</td><td>> 4 000</td></tr>
    <tr><td>Oxygène dissous (mg/L)</td><td>> 6</td><td>4 \u00e0 6</td><td>2 \u00e0 4</td><td>< 2</td></tr>
    <tr><td rowspan="5">\U0001f9ea Chimique</td><td>Nitrates (mg/L)</td><td>< 50</td><td>50 \u00e0 80</td><td>80 \u00e0 150</td><td>> 150</td></tr>
    <tr><td>Nitrites (mg/L)</td><td>< 3</td><td>3 \u00e0 5</td><td>5 \u00e0 10</td><td>> 10</td></tr>
    <tr><td>Ammonium (mg/L)</td><td>< 1,5</td><td>1,5 \u00e0 3</td><td>3 \u00e0 5</td><td>> 5</td></tr>
    <tr><td>Plomb (mg/L)</td><td>< 0,01</td><td>0,01 \u00e0 0,02</td><td>0,02 \u00e0 0,05</td><td>> 0,05</td></tr>
    <tr><td>Chlore résiduel (mg/L)</td><td>0,2 \u00e0 0,5</td><td>0,05 \u00e0 0,2</td><td>< 0,05</td><td>0 (non traité)</td></tr>
    </table>""", unsafe_allow_html=True)
    st.caption("Sources : OMS 2017, Norme béninoise NB 001/2001, USEPA 2022, Mama et al. 2011, SONEB bulletins qualité")

# ── PROTOCOLES ────────────────────────────────────────────────
with st.expander("\U0001f52c Protocoles de mesure \u2014 11 param\u00e8tres"):
    protos=[
        ("\U0001f9eb E.\u00a0coli \u2014 Test Colilert ou milieu m-FC (Microbiologie)",
         ["\U0001f527 Outil\u00a0: Kit Colilert (IDEXX) ou bandelettes Compact Dry EC",
          "1. Pr\u00e9lever 100\u00a0mL dans un flacon st\u00e9rile",
          "2. Ajouter le r\u00e9actif Colilert, m\u00e9langer jusqu\u2019\u00e0 dissolution",
          "3. Incuber \u00e0 35\u00a0\u00b0C pendant 24 \u00e0 28\u00a0heures",
          "4. Fluorescence UV = pr\u00e9sence d\u2019E.\u00a0coli \u2192 exprimer en UFC/100\u00a0mL"]),
        ("\U0001f9ea pH \u2014 Potentiom\u00e8tre (Physico-chimique)",
         ["\U0001f527 Outil\u00a0: pH-m\u00e8tre num\u00e9rique calibr\u00e9",
          "1. \u00c9talonner avec tampons pH\u00a04, pH\u00a07, pH\u00a010",
          "2. Plonger l\u2019\u00e9lectrode dans l\u2019\u00e9chantillon",
          "3. Attendre la stabilisation (30\u00a0secondes) et noter la valeur"]),
        ("\U0001f30a Turbidit\u00e9 \u2014 Turbidim\u00e8tre NTU (Physico-chimique)",
         ["\U0001f527 Outil\u00a0: Turbidim\u00e8tre num\u00e9rique",
          "1. Remplir le tube avec l\u2019\u00e9chantillon, essuyer le tube",
          "2. Ins\u00e9rer dans l\u2019appareil, fermer et lire la valeur NTU",
          "3. R\u00e9p\u00e9ter 3\u00a0fois et calculer la moyenne"]),
        ("\U0001f321\ufe0f Temp\u00e9rature \u2014 Thermom\u00e8tre sonde (Physico-chimique)",
         ["\U0001f527 Outil\u00a0: Thermom\u00e8tre num\u00e9rique immersible (pr\u00e9cision 0,1\u00a0\u00b0C)",
          "1. Plonger directement dans l\u2019\u00e9chantillon",
          "2. Attendre 1 \u00e0 2\u00a0min et lire la temp\u00e9rature en \u00b0C"]),
        ("\u26a1 Conductivit\u00e9 \u2014 Conductim\u00e8tre (Physico-chimique)",
         ["\U0001f527 Outil\u00a0: Conductim\u00e8tre portable \u00e9talonn\u00e9",
          "1. Rincer la cellule 2\u00a0fois avec l\u2019\u00e9chantillon",
          "2. Plonger, attendre stabilisation et lire en \u00b5S/cm"]),
        ("\U0001f4a8 Oxyg\u00e8ne dissous \u2014 Oxym\u00e8tre (Physico-chimique)",
         ["\U0001f527 Outil\u00a0: Oxym\u00e8tre num\u00e9rique avec sonde \u00e0 membrane",
          "1. \u00c9talonner dans l\u2019air satur\u00e9 en humidit\u00e9 (10\u00a0min)",
          "2. Plonger sans bulles, agiter doucement, lire apr\u00e8s 1\u00a02\u00a0min en mg/L"]),
        ("\U0001f33f Nitrates \u2014 Spectrophotom\u00e8tre ou bandelettes (Chimique)",
         ["\U0001f527 Outil\u00a0: Kit colorim\u00e9trique ou spectrophotom\u00e8tre (220\u00a0nm)",
          "1. Filtrer l\u2019\u00e9chantillon (0,45\u00a0\u00b5m)",
          "2. Bandelette\u00a0: tremper 1\u00a0s, attendre 60\u00a0s, comparer \u00e0 la charte"]),
        ("\U0001f9ea Nitrites \u2014 M\u00e9thode colorim\u00e9trique (Chimique)",
         ["\U0001f527 Outil\u00a0: Spectrophotom\u00e8tre (543\u00a0nm) ou kit Hach/Merck",
          "1. Ajouter le r\u00e9actif diazotation \u00e0 l\u2019\u00e9chantillon filtr\u00e9",
          "2. Attendre 10\u00a0min, lire l\u2019absorbance et exprimer en mg/L"]),
        ("\U0001f7e1 Ammonium \u2014 M\u00e9thode indoph\u00e9nol (Chimique)",
         ["\U0001f527 Outil\u00a0: Spectrophotom\u00e8tre (640\u00a0nm) ou kit rapide NH\u2084",
          "1. Ajouter les r\u00e9actifs (salicylate + hypochlorite) \u00e0 l\u2019\u00e9chantillon",
          "2. Attendre 30\u00a0min, lire \u00e0 640\u00a0nm, exprimer en mg/L NH\u2084"]),
        ("\U0001f4ab Plomb \u2014 Spectrophotom\u00e8trie d\u2019absorption atomique (Chimique)",
         ["\U0001f527 Outil\u00a0: Kit rapide Hach LCK309 ou spectrom\u00e8tre d\u2019absorption atomique (SAA)",
          "1. Acidifier l\u2019\u00e9chantillon (pH < 2) avec HNO\u2083 suppur pur",
          "2. Lire la concentration Pb en mg/L (SAA) ou comparer (kit rapide)"]),
        ("\U0001f4a7 Chlore r\u00e9siduel \u2014 M\u00e9thode DPD (Chimique)",
         ["\U0001f527 Outil\u00a0: Colorim\u00e8tre DPD ou chlorim\u00e8tre num\u00e9rique",
          "1. Ajouter la tablette DPD-1 \u00e0 l\u2019\u00e9chantillon (10\u00a0mL)",
          "2. Agiter jusqu\u2019\u00e0 dissolution, lire imm\u00e9diatement en mg/L Cl\u2082"]),
    ]
    for titre_p, items_p in protos:
        items_html="".join([f'<span class="proto-item">{i}</span>' for i in items_p])
        st.markdown(f'<div class="proto-box"><span class="proto-title">{titre_p}</span>{items_html}</div>',unsafe_allow_html=True)

# ── ANALYSTE & LIEU ───────────────────────────────────────────
st.markdown('<span class="section-title">\U0001f464 Informations sur l\u2019analyste et le pr\u00e9l\u00e8vement</span>',unsafe_allow_html=True)
analyste = st.text_input("Nom complet de l\u2019analyste *", placeholder="Ex\u00a0: Jean KOFFI")
lieu     = st.text_input("Lieu de pr\u00e9l\u00e8vement *", placeholder="Ex\u00a0: Village de Kpanr\u00f4u, commune de Djougou")
SOURCES  = ["Robinet (r\u00e9seau trait\u00e9 SONEB)","Puits peu profond",
            "Forage profond","Rivi\u00e8re","Fleuve / marigot","Lac","Eau stagnante (mare)",
            "Eau de pluie collect\u00e9e","Source naturelle","Eau de barrage",
            "Eau de citerne stock\u00e9e","Eau de mer / c\u00f4ti\u00e8re","Ros\u00e9e collect\u00e9e","Autre"]
source = st.selectbox("Source de l\u2019eau *", SOURCES)
cg1,cg2 = st.columns(2)
lat_input = cg1.number_input("Latitude (cartographie)", value=6.3703, step=0.0001, format="%.4f")
lon_input = cg2.number_input("Longitude (cartographie)", value=2.4305, step=0.0001, format="%.4f")

# ── SAISIE DES PARAMÈTRES ─────────────────────────────────────
st.markdown('<span class="section-title">\U0001f52c Saisie des mesures \u2014 trois mesures par param\u00e8tre</span>',unsafe_allow_html=True)
st.info("\u2139\ufe0f Pour chaque param\u00e8tre, saisissez les 3 mesures ou cochez \u00ab Pas mesur\u00e9 \u00bb. La moyenne est calcul\u00e9e automatiquement.")

def saisie_triple(cle, label_card, cat_class, ptext, pnorm, min_v, max_v, default, step, unite=""):
    st.markdown(f'<div class="pcard"><span class="plabel">{label_card}</span>'
                f'<span class="ptext">{ptext}</span>'
                f'<span class="pnorm">{pnorm}</span></div>', unsafe_allow_html=True)
    pas_mesure = st.checkbox(f"\U0001f6ab Pas mesur\u00e9 \u2014 {label_card.split('\u2014')[0].strip()}", key=f"pm_{cle}")
    if pas_mesure:
        st.caption("\U0001f7e0 Ce param\u00e8tre sera indiqu\u00e9 comme Non mesur\u00e9 dans le rapport.")
        return None
    st.markdown('<div class="mesure-group">', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    v1=c1.number_input(f"Mesure\u00a01",min_value=min_v,max_value=max_v,value=default,step=step,key=f"{cle}_1",label_visibility="visible")
    v2=c2.number_input(f"Mesure\u00a02",min_value=min_v,max_value=max_v,value=default,step=step,key=f"{cle}_2",label_visibility="visible")
    v3=c3.number_input(f"Mesure\u00a03",min_value=min_v,max_value=max_v,value=default,step=step,key=f"{cle}_3",label_visibility="visible")
    st.markdown('</div>', unsafe_allow_html=True)
    moy = round((v1+v2+v3)/3,5)
    st.caption(f"\U0001f4ca Moyenne {label_card.split('\u2014')[0].strip()}\u00a0: **{moy}** {unite}")
    return moy

# MICROBIOLOGIE
st.markdown('<div class="cat-micro"><span class="cat-title cat-title-micro">\U0001f9eb Qualit\u00e9 microbiologique</span>', unsafe_allow_html=True)
ecoli=saisie_triple("ecoli","E.\u00a0coli \u2014 Contamination f\u00e9cale","cat-micro",
    "Indicateur de contamination f\u00e9cale directe. Toute pr\u00e9sence = risque sanitaire.","Norme OMS : 0 UFC/100\u00a0mL",0.0,10000.0,0.0,1.0,"UFC/100\u00a0mL")
st.markdown('</div>', unsafe_allow_html=True)

# PHYSICO-CHIMIQUE
st.markdown('<div class="cat-physico"><span class="cat-title cat-title-physico">\u26a1 Qualit\u00e9 physico-chimique</span>', unsafe_allow_html=True)
pH_v=saisie_triple("pH","pH \u2014 Acidit\u00e9 / Basicit\u00e9","cat-physico",
    "pH bas\u00a0: m\u00e9taux toxiques. pH \u00e9lev\u00e9\u00a0: contamination chimique ou min\u00e9rale.","Norme OMS : 6,5 \u00e0 8,5",0.0,14.0,7.0,0.01)
turb=saisie_triple("turb","Turbidit\u00e9 (NTU) \u2014 Trouble de l\u2019eau","cat-physico",
    "Particules en suspension. Eau trouble = agents pathog\u00e8nes possibles, d\u00e9sinfection moins efficace.","Norme OMS : < 5 NTU",0.0,200.0,2.0,0.01,"NTU")
temp=saisie_triple("temp","Temp\u00e9rature (\u00b0C) \u2014 Activit\u00e9 microbienne","cat-physico",
    "Au-del\u00e0 de 25\u00a0\u00b0C, la prolif\u00e9ration des pathog\u00e8nes s\u2019acc\u00e9l\u00e8re.","Norme OMS : < 25\u00a0\u00b0C",0.0,60.0,27.0,0.1,"\u00b0C")
cond=saisie_triple("cond","Conductivit\u00e9 (\u00b5S/cm) \u2014 Min\u00e9ralisation","cat-physico",
    "Sels dissous excessifs = risques r\u00e9naux \u00e0 long terme.","Norme OMS : < 2\u202f500 \u00b5S/cm",0.0,10000.0,400.0,1.0,"\u00b5S/cm")
o2_v=saisie_triple("o2","Oxyg\u00e8ne dissous (mg/L) \u2014 Vitalit\u00e9","cat-physico",
    "Faible = d\u00e9composition organique, bact\u00e9ries. < 2\u00a0mg/L = eau anoxique.","Norme : > 6 mg/L",0.0,14.0,7.0,0.01,"mg/L")
st.markdown('</div>', unsafe_allow_html=True)

# CHIMIQUE
st.markdown('<div class="cat-chimique"><span class="cat-title cat-title-chimique">\U0001f9ea Qualit\u00e9 chimique</span>', unsafe_allow_html=True)
no3=saisie_triple("no3","Nitrates (mg/L) \u2014 Pollution agricole","cat-chimique",
    "Engrais agricoles. > 50\u00a0mg/L = m\u00e9th\u00e9moglobin\u00e9mie chez les nourrissons.","Norme OMS : < 50 mg/L",0.0,500.0,5.0,0.1,"mg/L")
no2=saisie_triple("no2","Nitrites (mg/L) \u2014 Contamination r\u00e9cente","cat-chimique",
    "Interm\u00e9diaire de nitrification. Indicateur de contamination organique r\u00e9cente.","Norme OMS : < 3 mg/L",0.0,20.0,0.01,0.001,"mg/L")
nh4=saisie_triple("nh4","Ammonium (mg/L) \u2014 D\u00e9gradation organique","cat-chimique",
    "Produit de d\u00e9gradation des mati\u00e8res organiques. Indicateur de contamination f\u00e9cale.","Norme OMS : < 1,5 mg/L",0.0,50.0,0.1,0.01,"mg/L")
pb_v=saisie_triple("pb","Plomb (mg/L) \u2014 M\u00e9tal lourd neurotoxique","cat-chimique",
    "Aucun seuil s\u00fbr. Neurotoxique. Dangereux m\u00eame \u00e0 tr\u00e8s faible dose pour les enfants.","Norme OMS : < 0,01 mg/L",0.0,1.0,0.002,0.0001,"mg/L")
cl_v=saisie_triple("cl","Chlore r\u00e9siduel (mg/L) \u2014 D\u00e9sinfectant","cat-chimique",
    "D\u00e9sinfectant r\u00e9siduel pour eau trait\u00e9e. Absent pour sources naturelles non trait\u00e9es.","Cible : 0,2 \u00e0 0,5 mg/L",0.0,5.0,0.0,0.001,"mg/L")
st.markdown('</div>', unsafe_allow_html=True)

# Indice de pollution chimique
if no3 is not None and no2 is not None and nh4 is not None:
    pi = round(no3 + no2*10 + nh4, 2)
    ni = "\u2705 Faible (<10)" if pi<10 else ("\u26a0\ufe0f Mod\u00e9r\u00e9 (10-50)" if pi<50 else ("\u274c \u00c9lev\u00e9 (50-150)" if pi<150 else "\u2620\ufe0f Critique (>150)"))
    st.markdown(f'<div class="pi-box">\U0001f4ca <b>Indice de pollution chimique</b> (NO\u2083 + NO\u2082\u00d710 + NH\u2084) = <b>{pi}</b> \u2192 {ni}</div>',unsafe_allow_html=True)

st.markdown("---")
MP={0:("\U0001f4a7 POTABLE","potable","Eau conforme aux normes OMS."),
    1:("\u26a0\ufe0f DOUTEUSE","douteuse","Anomalies d\u00e9tect\u00e9es. Filtrer et bouillir avant consommation."),
    2:("\u274c POLLU\u00c9E","polluee","Eau pollu\u00e9e. Traitement obligatoire."),
    3:("\u2620\ufe0f DANGEREUSE","dangereuse","DANGER EXTR\u00caM E. Tout contact \u00e0 \u00e9viter.")}

if st.button("\U0001f50d Analyser la qualit\u00e9 de l\u2019eau"):
    erreurs=[]
    if not analyste.strip(): erreurs.append("Le nom de l\u2019analyste est obligatoire.")
    if not lieu.strip():     erreurs.append("Le lieu de pr\u00e9l\u00e8vement est obligatoire.")
    # Vérifier au moins 3 paramètres mesurés
    vals_saisies={'ecoli':ecoli,'pH':pH_v,'turb':turb,'temp':temp,'cond':cond,
                  'o2':o2_v,'no3':no3,'no2':no2,'nh4':nh4,'pb':pb_v,'cl':cl_v}
    nb_mesures=sum(1 for v in vals_saisies.values() if v is not None)
    if nb_mesures < 3:
        erreurs.append("Au moins 3 param\u00e8tres doivent \u00eatre mesur\u00e9s pour effectuer l\u2019analyse.")
    if erreurs:
        for e in erreurs: st.error(e)
    else:
        # Préparer les features pour le modèle (remplacer None par défauts)
        feat_model={'Ecoli':ecoli,'pH':pH_v,'Turbidite':turb,'Temperature':temp,
                    'Conductivite':cond,'O2':o2_v,'Nitrates':no3,'Nitrites':no2,
                    'Ammonium':nh4,'Plomb':pb_v,'Chlore':cl_v}
        cl_pred, pr = predict_with_missing(feat_model)
        lb,cs,co_msg=MP[cl_pred]

        # Évaluer sous réserve
        label_final, sous_reserve, params_manquants = evaluer_sous_reserve(vals_saisies, cl_pred)

        # Stocker dans session_state
        st.session_state.analyse_faite=True
        st.session_state.dernier_resultat={
            "lb":lb,"cs":cs,"co_msg":co_msg,"cl":int(cl_pred),"pr":list(pr),
            "conf":round(pr[cl_pred]*100,1),"label_final":label_final,
            "sous_reserve":sous_reserve,"params_manquants":params_manquants,
            "vals":vals_saisies,"analyste":analyste,"lieu":lieu,"source":source,
            "lat":lat_input,"lon":lon_input,
        }
        # Générer PDF
        try:
            pdf_bytes=generer_pdf(vals_saisies,cl_pred,list(pr),analyste,lieu,source,
                                  label_final,sous_reserve,params_manquants)
            st.session_state.dernier_pdf=pdf_bytes
            st.session_state.dernier_pdf_nom="rapport_eauvie_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".pdf"
        except Exception as e:
            st.session_state.dernier_pdf=None
            st.error("Erreur PDF\u00a0: "+str(e))
        # Historique
        st.session_state.histo.append({
            "Heure":datetime.now().strftime("%H:%M:%S"),"Analyste":analyste,
            "Lieu":lieu,"Source":source,"R\u00e9sultat":label_final,
            "E.coli":"NM" if ecoli is None else ecoli,
            "pH":"NM" if pH_v is None else pH_v,
            "Turb.":"NM" if turb is None else turb,
            "Temp.":"NM" if temp is None else temp,
            "Cond.":"NM" if cond is None else cond,
            "O2":"NM" if o2_v is None else o2_v,
            "NO3":"NM" if no3 is None else no3,
            "NO2":"NM" if no2 is None else no2,
            "NH4":"NM" if nh4 is None else nh4,
            "Pb":"NM" if pb_v is None else pb_v,
            "Cl":"NM" if cl_v is None else cl_v,
        })

# ── RÉSULTATS PERSISTANTS ─────────────────────────────────────
if st.session_state.analyse_faite and st.session_state.dernier_resultat:
    r=st.session_state.dernier_resultat
    lb=r["lb"]; cs=r["cs"]; label_final=r["label_final"]
    cl=r["cl"]; pr=r["pr"]; conf=r["conf"]
    sous_reserve=r["sous_reserve"]; params_manquants=r["params_manquants"]

    st.markdown(f'<div class="result-box {cs}">{label_final}<br>'
                f'<span style="font-size:13px;font-weight:600;">Confiance du mod\u00e8le\u00a0: {conf}\u00a0%</span></div>',
                unsafe_allow_html=True)

    # Alerte sous réserve
    if sous_reserve and params_manquants:
        manq_str=", ".join(params_manquants)
        st.markdown(f'<div class="sous-reserve-box">\u26a0\ufe0f <b>Param\u00e8tres critiques non mesur\u00e9s :</b> {manq_str}<br/>'
                    f'La d\u00e9cision est provisoire. Ajoutez ces mesures pour une confirmation d\u00e9finitive.</div>',
                    unsafe_allow_html=True)

    # Indice pollution affiché dans les résultats
    no3_r=r["vals"].get("no3"); no2_r=r["vals"].get("no2"); nh4_r=r["vals"].get("nh4")
    if no3_r is not None and no2_r is not None and nh4_r is not None:
        pi_r=round(no3_r+no2_r*10+nh4_r,2)
        ni_r="\u2705 Faible" if pi_r<10 else ("\u26a0\ufe0f Mod\u00e9r\u00e9" if pi_r<50 else ("\u274c \u00c9lev\u00e9" if pi_r<150 else "\u2620\ufe0f Critique"))
        st.markdown(f'<div class="pi-box"><b>Indice de pollution chimique :</b> {pi_r} \u2192 {ni_r}</div>',unsafe_allow_html=True)

    st.markdown(f"**\U0001f4a1 Conseil\u00a0:** {r['co_msg']}")

    if cl in [1,2,3]:
        with st.expander("\U0001f6e0\ufe0f M\u00e9thodes de purification recommand\u00e9es"):
            for tc,dc in [("\U0001f525 1. \u00c9bullition","5\u00a0min minimum. Efficace contre bact\u00e9ries, virus, parasites."),
                          ("\U0001f9f4 2. Filtration sable/gravier","Gravier + sable + charbon actif. \u00c0 compl\u00e9ter par \u00e9bullition."),
                          ("\u2600\ufe0f 3. SODIS","Bouteilles transparentes, 6\u00a0h soleil (ou 2\u00a0j nuageux). Valid\u00e9 OMS."),
                          ("\U0001f9ea 4. Chloration","2\u00a0gouttes Javel 5\u00a0% / litre. Attendre 30\u00a0min."),
                          ("\U0001f331 5. Moringa oleifera","2-3\u00a0graines broy\u00e9es dans 1\u00a0L. Agiter + d\u00e9canter 1\u00a0h.")]:
                st.markdown(f'<div class="conseil-box"><span class="conseil-title">{tc}</span><span class="conseil-item">{dc}</span></div>',unsafe_allow_html=True)

    # Graphique
    prd=pd.DataFrame({"Classe":["Potable","Douteuse","Pollu\u00e9e","Dangereuse"],
                       "Probabilit\u00e9 (%)":[round(p*100,1) for p in pr]})
    st.bar_chart(prd.set_index("Classe"))

    # PDF
    st.markdown("---")
    st.markdown('<div class="pdf-box">\U0001f4c4 <b>Rapport PDF officiel \u2014 11 param\u00e8tres \u2014 3 cat\u00e9gories</b></div>',unsafe_allow_html=True)
    if st.session_state.dernier_pdf:
        st.download_button("\U0001f4e5 T\u00e9l\u00e9charger le rapport PDF",
            data=st.session_state.dernier_pdf,
            file_name=st.session_state.dernier_pdf_nom,
            mime="application/pdf",key="dl_pdf")
        st.success("\u2705 Rapport pr\u00eat.")

    # Cartographie
    st.markdown("---")
    st.markdown('<div class="carto-box">\U0001f30d <b>Cartographie communautaire</b><br>'
                'Si votre mesure est r\u00e9elle, ajoutez-la \u00e0 la carte pour faciliter les d\u00e9cisions des autorit\u00e9s et des ONG.</div>',
                unsafe_allow_html=True)
    if st.button("\U0001f4cd Ajouter \u00e0 la cartographie", key="btn_carto"):
        point={"lat":r["lat"],"lon":r["lon"],"lieu":r["lieu"],"source":r["source"],
               "resultat":label_final,"classe":r["cl"],"analyste":r["analyste"],
               "date":datetime.now().strftime("%d/%m/%Y"),"heure":datetime.now().strftime("%H:%M"),
               **{k:(round(v,4) if v is not None else "NM") for k,v in r["vals"].items()}}
        st.session_state.carto_points.append(point)
        st.success(f"\u2705 Ajout\u00e9 ({len(st.session_state.carto_points)} point(s)).")

# ── HISTORIQUE ────────────────────────────────────────────────
if len(st.session_state.histo)>0:
    st.markdown("---")
    st.markdown('<span class="section-title">\U0001f554 Historique des analyses</span>',unsafe_allow_html=True)
    hdf=pd.DataFrame(st.session_state.histo)
    st.dataframe(hdf,use_container_width=True)
    st.download_button("\u2b07\ufe0f CSV",hdf.to_csv(index=False).encode("utf-8"),"historique_eauvie.csv","text/csv",key="dl_csv")

# ── CARTOGRAPHIE ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<span class="section-title">\U0001f5fa\ufe0f Cartographie des analyses</span>',unsafe_allow_html=True)
mdp=st.text_input("\U0001f512 Mot de passe",type="password",placeholder="Saisir le mot de passe",key="mdp_c")
if mdp:
    if mdp=="CARTOGRAPHIE":
        pts=st.session_state.carto_points; nb=len(pts)
        st.success(f"\u2705 Acc\u00e8s autoris\u00e9. {nb} point(s).")
        if nb==0:
            st.info("\U0001f4cd Aucune mesure ajout\u00e9e. Effectuez une analyse et cliquez sur \u00ab Ajouter \u00e0 la cartographie \u00bb.")
        else:
            try:
                import folium; from streamlit_folium import st_folium
                clat=sum(p["lat"] for p in pts)/nb; clon=sum(p["lon"] for p in pts)/nb
                m=folium.Map(location=[clat,clon],zoom_start=8,tiles="CartoDB positron")
                COUL={0:"green",1:"orange",2:"red",3:"darkred"}
                for p in pts:
                    ph=folium.Popup(f"<b>{p['resultat']}</b><br>Lieu: {p['lieu']}<br>Source: {p['source']}<br>Analyste: {p['analyste']}<br>{p['date']} \u00e0 {p['heure']}",max_width=250)
                    folium.Marker([p["lat"],p["lon"]],popup=ph,
                        tooltip=f"{p['resultat']} \u2014 {p['lieu']}",
                        icon=folium.Icon(color=COUL.get(p["classe"],"blue"),icon="info-sign",prefix="glyphicon")).add_to(m)
                st_folium(m,width=700,height=420)
                df_c=pd.DataFrame(pts); st.dataframe(df_c,use_container_width=True)
                ca,cb=st.columns(2)
                ca.download_button("\u2b07\ufe0f CSV",df_c.to_csv(index=False).encode("utf-8"),"carto_eauvie.csv","text/csv",key="dl_cc")
                cb.download_button("\u2b07\ufe0f JSON",json.dumps({"points":pts},ensure_ascii=False,indent=2).encode("utf-8"),"carto_eauvie.json","application/json",key="dl_cj")
            except ImportError:
                st.dataframe(pd.DataFrame(pts),use_container_width=True)
    else:
        st.error("\u274c Mot de passe incorrect.")

# ── CONTACT ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<span class="section-title">\U0001f4e7 Contacter le d\u00e9veloppeur</span>',unsafe_allow_html=True)
st.markdown("""<div style="background:linear-gradient(135deg,#023e8a,#0077b6);border-radius:14px;padding:18px;text-align:center;">
<div style="color:#fff;font-size:15px;font-weight:800;margin-bottom:6px;">\U0001f4e7 Charles MEDEZOUNDJI</div>
<div style="color:#d0eeff;font-size:12px;margin-bottom:14px;">D\u00e9veloppeur d\u2019EauVie \u2014 B\u00e9nin, Afrique de l\u2019Ouest</div>
<a href="mailto:charlesezechielmedezoundji@gmail.com?subject=EauVie%20-%20Message&body=Bonjour%20Charles%2C%0A%0AJe%20vous%20contacte%20au%20sujet%20d'EauVie.%0A%0A"
   target="_blank"
   style="display:inline-block;background:white;color:#023e8a;font-weight:800;font-size:14px;padding:10px 26px;border-radius:10px;text-decoration:none;">
\U0001f4e4 Envoyer un message
</a>
<div style="color:#a8d8ff;font-size:10px;margin-top:10px;">charlesezechielmedezoundji@gmail.com</div>
</div>""", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div style="text-align:center;color:#023e8a !important;font-size:11px;padding:8px;font-weight:600;">\U0001f4a7 EauVie \u2014 11 param\u00e8tres \u2014 3 cat\u00e9gories \u2014 Random Forest + Feature Engineering \u2014 Normes OMS \u2014 Charles MEDEZOUNDJI \u2014 2026</div>',unsafe_allow_html=True)
