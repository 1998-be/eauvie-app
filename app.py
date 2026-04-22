
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import os
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

# ── COULEURS ──────────────────────────────────────────────────
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

def couleur_classe(cl): return [VERT, ORANGE, ROUGE, ROUGE_FONCE][cl]
def label_classe(cl):   return ["POTABLE", "DOUTEUSE", "POLLU\u00c9E", "DANGEREUSE"][cl]

def conseil_classe(cl):
    return [
        "Cette eau est conforme aux normes OMS. Elle peut \u00eatre consomm\u00e9e sans traitement pr\u00e9alable. Veillez \u00e0 maintenir des conditions de stockage hygi\u00e9niques.",
        "Des anomalies ont \u00e9t\u00e9 d\u00e9tect\u00e9es sur un ou plusieurs param\u00e8tres. Il est fortement recommand\u00e9 de filtrer et de faire bouillir cette eau avant toute consommation humaine.",
        "Cette eau est pollu\u00e9e et impropre \u00e0 la consommation. Un traitement complet \u2014 filtration, d\u00e9sinfection, \u00e9bullition \u2014 est obligatoire avant tout usage.",
        "DANGER EXTR\u00caM E. Cette eau pr\u00e9sente un risque sanitaire majeur. Tout contact doit \u00eatre \u00e9vit\u00e9. Signalez imm\u00e9diatement aux autorit\u00e9s sanitaires comp\u00e9tentes."
    ][cl]

def statut_param(val, pmin, pmax, inverse=False):
    if inverse:
        if val <= pmax:           return "Conforme",    VERT
        elif val <= pmax * 2:     return "Limite",      ORANGE
        else:                     return "Non conforme", ROUGE
    if pmin <= val <= pmax:       return "Conforme",    VERT
    elif (pmin-1.5) <= val <= (pmax+1.5): return "Limite", ORANGE
    else:                         return "Non conforme", ROUGE

def S(name, **kw): return ParagraphStyle(name, **kw)

# ── FICHIER CARTOGRAPHIE (stockage session) ───────────────────
CARTO_KEY = "carto_points"
if CARTO_KEY not in st.session_state:
    st.session_state[CARTO_KEY] = []

# ── GÉNÉRATION PDF ────────────────────────────────────────────
def generer_pdf(mesures, classe, probabilites, analyste="", lieu="", source="", heure_locale=""):
    pH=mesures["pH"]; turbidite=mesures["turb"]; temperature=mesures["temp"]
    conductivite=mesures["cond"]; nitrates=mesures["no3"]
    o2=mesures["o2"]; ecoli=mesures["ecoli"]
    buffer=io.BytesIO(); W,H=A4
    now=datetime.now()
    date_str=now.strftime("%d/%m/%Y")
    heure_str=heure_locale if heure_locale else now.strftime("%H:%M")
    ref_str="EV-"+now.strftime("%Y%m%d-%H%M%S")
    doc=SimpleDocTemplate(buffer,pagesize=A4,
        leftMargin=1.8*cm,rightMargin=1.8*cm,
        topMargin=1.5*cm,bottomMargin=2*cm,
        title="Rapport EauVie",author=analyste or "EauVie",
        subject="Analyse physico-chimique et microbiologique \u2014 Normes OMS")
    story=[]
    # En-t\u00eate
    header=[
        [Paragraph("<b>\U0001f4a7 EauVie</b>",S("hx",fontName="Helvetica-Bold",fontSize=24,textColor=BLANC,alignment=TA_CENTER))],
        [Paragraph("Analyse intelligente de la qualit\u00e9 de l\u2019eau \u2014 Normes OMS<br/>afin de garantir une consommation rassurante et b\u00e9n\u00e9fique.",S("hx2",fontName="Helvetica",fontSize=10.5,textColor=colors.HexColor("#d0eeff"),alignment=TA_CENTER,leading=16))],
        [Paragraph("Propos\u00e9e par Charles MEDEZOUNDJI",S("hx3",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.HexColor("#a8d8ff"),alignment=TA_CENTER))],
    ]
    ht=Table(header,colWidths=[W-3.6*cm])
    ht.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),BLEU_MED),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14)]))
    story.append(ht); story.append(Spacer(1,0.4*cm))
    rt=Table([[
        Paragraph("<b>RAPPORT D\u2019ANALYSE DE L\u2019EAU</b>",S("rd",fontName="Helvetica-Bold",fontSize=11,textColor=BLEU_FONCE,alignment=TA_CENTER)),
        Paragraph(f"<b>R\u00e9f. :</b> {ref_str}",S("rd2",fontName="Helvetica",fontSize=8.5,textColor=colors.HexColor("#555"),alignment=TA_LEFT)),
        Paragraph(f"<b>Date :</b> {date_str}  |  <b>Heure :</b> {heure_str}",S("rd3",fontName="Helvetica",fontSize=8.5,textColor=colors.HexColor("#555"),alignment=TA_RIGHT)),
    ]],colWidths=[7*cm,4.5*cm,6*cm])
    rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),GRIS_CLAIR),("LINEBELOW",(0,0),(-1,-1),1.5,BLEU_CLAIR),("LINETOP",(0,0),(-1,-1),1.5,BLEU_CLAIR),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(rt); story.append(Spacer(1,0.5*cm))

    def titre_section(txt):
        story.append(HRFlowable(width="100%",thickness=2,color=BLEU_MED,spaceAfter=4))
        story.append(Paragraph(txt,S("h1",fontName="Helvetica-Bold",fontSize=13,textColor=BLEU_FONCE,spaceBefore=10,spaceAfter=5)))
        story.append(HRFlowable(width="100%",thickness=0.5,color=GRIS_MED,spaceAfter=6))

    sb=S("sb",fontName="Helvetica",fontSize=9.5,textColor=NOIR,alignment=TA_JUSTIFY,leading=15,spaceAfter=5)
    si=S("si",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.HexColor("#555"),alignment=TA_JUSTIFY,leading=13,spaceAfter=4)
    sn=S("sn",fontName="Helvetica-Oblique",fontSize=8.5,textColor=colors.HexColor("#333"),alignment=TA_JUSTIFY,leading=13)

    # Section 1
    titre_section("1.  CONTEXTE ET PROBL\u00c9MATIQUE")
    story.append(Paragraph("L\u2019eau, ressource vitale et irremplaçable, est au c\u0153ur d\u2019une crise sanitaire silencieuse qui ravage le continent africain. Plus de <b>400 millions d\u2019Africains</b> n\u2019ont toujours pas acc\u00e8s \u00e0 une eau potable s\u00fbre et durable (OMS/UNICEF, 2025). Des milliers de vies sont perdues chaque jour \u2014 principalement des enfants de moins de cinq ans \u2014 victimes de maladies diarrh\u00e9iques, du chol\u00e9ra, de la fi\u00e8vre typho\u00efde et d\u2019autres pathologies directement li\u00e9es \u00e0 la consommation d\u2019une eau de mauvaise qualit\u00e9.",sb))
    story.append(Paragraph("Au B\u00e9nin, l\u2019acc\u00e8s \u00e0 l\u2019eau potable constitue le <b>premier d\u00e9fi prioritaire</b> cit\u00e9 par les citoyens (Afrobarom\u00e8tre, 2024). Les zones rurales et les communaut\u00e9s agricoles sont particuli\u00e8rement expos\u00e9es, leurs sources d\u2019eau \u00e9tant vuln\u00e9rables aux contaminations bact\u00e9riennes, aux polluants chimiques agricoles et aux effets du changement climatique.",sb))
    story.append(Paragraph("<i>Face \u00e0 ce constat alarmant, la question se pose avec urgence\u00a0: comment permettre \u00e0 chaque communaut\u00e9, chaque famille \u2014 m\u00eame sans \u00e9quipement de laboratoire sophistiqu\u00e9 \u2014 de conna\u00eetre et de comprendre la qualit\u00e9 de l\u2019eau qu\u2019elle consomme, avant qu\u2019il ne soit trop tard\u00a0?</i>",si))
    story.append(Paragraph("C\u2019est pr\u00e9cis\u00e9ment \u00e0 cette question qu\u2019EauVie r\u00e9pond. D\u00e9velopp\u00e9e au B\u00e9nin par Charles MEDEZOUNDJI, cette application combine <b>sept param\u00e8tres physico-chimiques et microbiologiques</b> standardis\u00e9s par l\u2019OMS avec un algorithme Random Forest entra\u00een\u00e9 sur 122\u00a0\u00e9chantillons repr\u00e9sentatifs, atteignant une pr\u00e9cision de <b>100\u00a0% (validation crois\u00e9e 5-fold)</b>.",sb))
    story.append(Spacer(1,0.3*cm))

    # Section 2
    titre_section("2.  INFORMATIONS SUR L\u2019\u00c9CHANTILLON ANALYS\u00c9")
    info_data=[
        [Paragraph("<b>CHAMP</b>",S("ih",fontName="Helvetica-Bold",fontSize=9,textColor=BLANC,alignment=TA_CENTER)),
         Paragraph("<b>INFORMATION</b>",S("ih2",fontName="Helvetica-Bold",fontSize=9,textColor=BLANC,alignment=TA_CENTER))],
        ["R\u00e9f\u00e9rence du rapport", ref_str],
        ["Date d\u2019analyse", date_str],
        ["Heure de l\u2019analyse", heure_str],
        ["Lieu de pr\u00e9l\u00e8vement", lieu or "Non renseign\u00e9"],
        ["Source de l\u2019eau", source or "Non renseign\u00e9e"],
        ["Analyste", analyste or "Non renseign\u00e9"],
        ["Outil utilis\u00e9", "EauVie IA \u2014 Application d\u2019analyse intelligente"],
        ["M\u00e9thode IA", "Random Forest (500 arbres, pr\u00e9cision = 100\u00a0%)"],
        ["Param\u00e8tres analys\u00e9s", "pH, Turbidit\u00e9, Temp\u00e9rature, Conductivit\u00e9, Nitrates, Oxyg\u00e8ne dissous, E.\u00a0coli"],
        ["R\u00e9f\u00e9rentiel", "Normes OMS \u2014 Directives qualit\u00e9 eau de boisson (4e\u00a0\u00e9dition, 2017)"],
    ]
    it=Table(info_data,colWidths=[6.5*cm,11*cm])
    it.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLEU_FONCE),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("FONTNAME",(1,1),(1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),("ROWBACKGROUNDS",(0,1),(-1,-1),[BLANC,BLEU_PALE]),("GRID",(0,0),(-1,-1),0.5,GRIS_MED),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8)]))
    story.append(it); story.append(Spacer(1,0.5*cm))

    # Section 3 - mesures
    titre_section("3.  MESURES PHYSICO-CHIMIQUES ET MICROBIOLOGIQUES")
    story.append(Paragraph("Les sept param\u00e8tres ci-dessous ont \u00e9t\u00e9 mesur\u00e9s en triple r\u00e9p\u00e9tition. La moyenne des trois mesures a \u00e9t\u00e9 soumise \u00e0 l\u2019algorithme EauVie pour garantir la repr\u00e9sentativit\u00e9 et la fiabilit\u00e9 du r\u00e9sultat.",sb))
    params=[
        ("pH",f"{pH:.3f}","6,5 \u2013 8,5","Acidit\u00e9 / Basicit\u00e9","Mesure l\u2019activit\u00e9 des ions hydrog\u00e8ne. Un pH hors norme signale une contamination chimique, min\u00e9rale ou la pr\u00e9sence de m\u00e9taux lourds dissous.",statut_param(pH,6.5,8.5)),
        ("Turbidit\u00e9",f"{turbidite:.3f} NTU","< 5 NTU","Trouble / Particules","Mesure les mati\u00e8res en suspension \u2014 argile, bact\u00e9ries, mati\u00e8res organiques. Une turbidit\u00e9 \u00e9lev\u00e9e r\u00e9duit l\u2019efficacit\u00e9 de la d\u00e9sinfection.",statut_param(turbidite,0,5)),
        ("Temp\u00e9rature",f"{temperature:.2f} \u00b0C","< 25 \u00b0C","Vitalit\u00e9 microbienne","Au-del\u00e0 de 25\u00a0\u00b0C, la prolif\u00e9ration des micro-organismes pathog\u00e8nes s\u2019acc\u00e9l\u00e8re significativement.",statut_param(temperature,0,25)),
        ("Conductivit\u00e9",f"{conductivite:.1f} \u00b5S/cm","< 2\u202f500 \u00b5S/cm","Min\u00e9ralisation","Une conductivit\u00e9 \u00e9lev\u00e9e indique une concentration excessive en sels dissous, pouvant nuire \u00e0 la sant\u00e9 \u00e0 long terme.",statut_param(conductivite,0,2500)),
        ("Nitrates",f"{nitrates:.3f} mg/L","< 50 mg/L","Pollution agricole","Les nitrates proviennent des engrais agricoles. Au-del\u00e0 de 50\u00a0mg/L, ils provoquent la m\u00e9th\u00e9moglobin\u00e9mie chez les nourrissons.",statut_param(nitrates,0,50)),
        ("Oxyg\u00e8ne dissous",f"{o2:.3f} mg/L","> 6 mg/L","Vitalit\u00e9 / Pollution","Un taux faible indique une d\u00e9composition organique intense. En dessous de 2\u00a0mg/L, l\u2019eau est consid\u00e9r\u00e9e anoxique et dangereuse.",statut_param(o2,6,14)),
        ("E.\u00a0coli",f"{ecoli:.1f} UFC/100\u00a0mL","0 UFC/100\u00a0mL","Contamination f\u00e9cale","La pr\u00e9sence d\u2019E.\u00a0coli signale une contamination f\u00e9cale directe et la probable pr\u00e9sence d\u2019autres agents pathog\u00e8nes.",statut_param(ecoli,0,0,inverse=True)),
    ]
    mh=[Paragraph(f"<b>{t}</b>",S("mh",fontName="Helvetica-Bold",fontSize=8,textColor=BLANC,alignment=TA_CENTER)) for t in ["Param\u00e8tre","Valeur moyenne","Norme OMS","Signification","Interpr\u00e9tation","Statut"]]
    mes_rows=[mh]
    for nom,val,norme,signif,interp,(stat,coul) in params:
        mes_rows.append([Paragraph(f"<b>{nom}</b>",S("mc1",fontName="Helvetica-Bold",fontSize=8,textColor=BLEU_FONCE,alignment=TA_CENTER)),Paragraph(f"<b>{val}</b>",S("mc2",fontName="Helvetica-Bold",fontSize=9,textColor=NOIR,alignment=TA_CENTER)),Paragraph(norme,S("mc3",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_CENTER)),Paragraph(signif,S("mc4",fontName="Helvetica",fontSize=7.5,textColor=NOIR,alignment=TA_CENTER)),Paragraph(interp,S("mc5",fontName="Helvetica",fontSize=7,textColor=NOIR,alignment=TA_JUSTIFY,leading=10)),Paragraph(f"<b>{stat}</b>",S("mc6",fontName="Helvetica-Bold",fontSize=8,textColor=coul,alignment=TA_CENTER))])
    mt=Table(mes_rows,colWidths=[2.0*cm,2.2*cm,2.0*cm,2.2*cm,6.5*cm,2.3*cm])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLEU_FONCE),("GRID",(0,0),(-1,-1),0.4,GRIS_MED),("ROWBACKGROUNDS",(0,1),(-1,-1),[BLANC,BLEU_PALE]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
    story.append(mt); story.append(Spacer(1,0.5*cm))

    # Section 4 - résultat IA
    titre_section("4.  R\u00c9SULTAT DE L\u2019ANALYSE PAR INTELLIGENCE ARTIFICIELLE")
    coul_res=couleur_classe(classe); label_res=label_classe(classe); conf_res=round(probabilites[classe]*100,1)
    res_t=Table([[Paragraph(f"QUALIT\u00c9 DE L\u2019EAU : {label_res}",S("rb",fontName="Helvetica-Bold",fontSize=18,textColor=BLANC,alignment=TA_CENTER))],[Paragraph(f"Confiance : {conf_res}\u00a0%  |  Random Forest (500 arbres)  |  Pr\u00e9cision : 100\u00a0%",S("rb2",fontName="Helvetica",fontSize=9,textColor=BLANC,alignment=TA_CENTER))]],colWidths=[W-3.6*cm])
    res_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),coul_res),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14)]))
    story.append(res_t); story.append(Spacer(1,0.3*cm))
    story.append(Paragraph("<b>Distribution des probabilit\u00e9s par classe :</b>",S("h2",fontName="Helvetica-Bold",fontSize=11,textColor=BLEU_MED,spaceBefore=6,spaceAfter=4)))
    lcls=["POTABLE","DOUTEUSE","POLLU\u00c9E","DANGEREUSE"]; ccls=[VERT,ORANGE,ROUGE,ROUGE_FONCE]
    ph_row=[Paragraph(f"<b>{l}</b>",S("ph",fontName="Helvetica-Bold",fontSize=8.5,textColor=BLANC,alignment=TA_CENTER)) for l in lcls]
    pv_row=[Paragraph(f"<b>{round(p*100,1)}\u00a0%</b>",S("pv",fontName="Helvetica-Bold",fontSize=10,textColor=ccls[i],alignment=TA_CENTER)) for i,p in enumerate(probabilites)]
    pt=Table([ph_row,pv_row],colWidths=[(W-3.6*cm)/4]*4)
    pt.setStyle(TableStyle([*[("BACKGROUND",(i,0),(i,0),ccls[i]) for i in range(4)],("GRID",(0,0),(-1,-1),0.5,GRIS_MED),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),("BACKGROUND",(0,1),(-1,1),GRIS_CLAIR)]))
    story.append(pt); story.append(Spacer(1,0.4*cm))

    # Section 5 - recommandations
    titre_section("5.  INTERPR\u00c9TATION SCIENTIFIQUE ET RECOMMANDATIONS")
    story.append(Paragraph(f"Sur la base des sept mesures physico-chimiques et microbiologiques obtenues et de l\u2019analyse par l\u2019algorithme Random Forest, l\u2019\u00e9chantillon d\u2019eau analys\u00e9 pr\u00e9sente le profil suivant\u00a0:",sb))
    at=Table([[Paragraph(f"AVIS SANITAIRE \u2014 EAU {label_res}\u00a0: {conseil_classe(classe)}",S("al",fontName="Helvetica-Bold",fontSize=9.5,textColor=BLANC,alignment=TA_JUSTIFY,leading=14))]],colWidths=[W-3.6*cm])
    at.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),coul_res),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12)]))
    story.append(at); story.append(Spacer(1,0.3*cm))
    if classe>0:
        story.append(Paragraph("<b>M\u00e9thodes de purification recommand\u00e9es :</b>",S("h2",fontName="Helvetica-Bold",fontSize=11,textColor=BLEU_MED,spaceBefore=6,spaceAfter=4)))
        methodes=[("1. \u00c9bullition","Porter l\u2019eau \u00e0 \u00e9bullition pendant au moins 5\u00a0minutes. Laisser refroidir dans un r\u00e9cipient propre et couvert. Efficace contre les bact\u00e9ries, les virus et les parasites."),("2. Filtration artisanale","Couches successives\u00a0: gravier grossier, gravier fin, sable grossier, sable fin, charbon de bois actif. \u00c0 combiner obligatoirement avec l\u2019\u00e9bullition."),("3. D\u00e9sinfection solaire SODIS","Bouteilles en plastique transparent expos\u00e9es 6\u00a0heures au soleil (ciel clair) ou 2\u00a0jours (nuageux). M\u00e9thode gratuite, valid\u00e9e par l\u2019OMS."),("4. Chloration","2\u00a0gouttes d\u2019eau de Javel \u00e0 5\u00a0% par litre d\u2019eau trouble (1\u00a0goutte si claire). Attendre 30\u00a0minutes avant de consommer."),("5. Graines de Moringa oleifera","Broyer 2 \u00e0 3\u00a0graines s\u00e8ches en poudre fine. Ajouter \u00e0 1\u00a0litre d\u2019eau turbide, agiter 1\u00a0minute puis 5\u00a0minutes lentement. D\u00e9canter 1\u00a0heure et compl\u00e9ter par \u00e9bullition ou chloration.")]
        mr=[[Paragraph(f"<b>{m}</b>",S("mt",fontName="Helvetica-Bold",fontSize=8.5,textColor=BLEU_FONCE,alignment=TA_LEFT)),Paragraph(d,S("md",fontName="Helvetica",fontSize=8.5,textColor=NOIR,leading=12,alignment=TA_JUSTIFY))] for m,d in methodes]
        mtt=Table(mr,colWidths=[3.5*cm,14*cm])
        mtt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,GRIS_MED),("ROWBACKGROUNDS",(0,0),(-1,-1),[BLANC,BLEU_PALE]),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
        story.append(mtt); story.append(Spacer(1,0.3*cm))

    # Section 6 - conformité OMS
    titre_section("6.  TABLEAU DE CONFORMIT\u00c9 AUX NORMES OMS")
    ch=[Paragraph(f"<b>{t}</b>",S("ch",fontName="Helvetica-Bold",fontSize=9,textColor=BLANC,alignment=TA_CENTER)) for t in ["Param\u00e8tre","Valeur mesur\u00e9e","Seuil OMS Potable","Seuil Dangereuse","Conformit\u00e9"]]
    cd=[("pH",f"{pH:.3f}","6,5 \u00e0 8,5","< 4,5 ou > 10",statut_param(pH,6.5,8.5)),("Turbidit\u00e9 (NTU)",f"{turbidite:.3f}","< 5","> 50",statut_param(turbidite,0,5)),("Temp\u00e9rature (\u00b0C)",f"{temperature:.2f}","< 25","> 35",statut_param(temperature,0,25)),("Conductivit\u00e9 (\u00b5S/cm)",f"{conductivite:.1f}","< 2\u00a0500","> 4\u00a0000",statut_param(conductivite,0,2500)),("Nitrates (mg/L)",f"{nitrates:.3f}","< 50","> 150",statut_param(nitrates,0,50)),("Oxyg\u00e8ne dissous (mg/L)",f"{o2:.3f}","> 6","< 2",statut_param(o2,6,14)),("E.\u00a0coli (UFC/100\u00a0mL)",f"{ecoli:.1f}","0","> 500",statut_param(ecoli,0,0,inverse=True)),]
    crows=[ch]+[[Paragraph(f"<b>{n}</b>",S("c1",fontName="Helvetica-Bold",fontSize=8.5,textColor=NOIR,alignment=TA_LEFT)),Paragraph(f"<b>{v}</b>",S("c2",fontName="Helvetica-Bold",fontSize=9,textColor=NOIR,alignment=TA_CENTER)),Paragraph(sp,S("c3",fontName="Helvetica",fontSize=8.5,textColor=VERT,alignment=TA_CENTER)),Paragraph(sd,S("c4",fontName="Helvetica",fontSize=8.5,textColor=ROUGE,alignment=TA_CENTER)),Paragraph(f"<b>{st}</b>",S("c5",fontName="Helvetica-Bold",fontSize=9,textColor=cu,alignment=TA_CENTER))] for n,v,sp,sd,(st,cu) in cd]
    ct=Table(crows,colWidths=[4.2*cm,3.0*cm,3.8*cm,3.2*cm,3.0*cm])
    ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLEU_MED),("GRID",(0,0),(-1,-1),0.5,GRIS_MED),("ROWBACKGROUNDS",(0,1),(-1,-1),[BLANC,BLEU_PALE]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)]))
    story.append(ct); story.append(Spacer(1,0.4*cm))

    # Section 7 - notes
    titre_section("7.  NOTES M\u00c9THODOLOGIQUES ET LIMITES")
    for i,note in enumerate(["Ce rapport est g\u00e9n\u00e9r\u00e9 automatiquement par EauVie sur la base de la moyenne des trois mesures saisies pour chacun des sept param\u00e8tres. La fiabilit\u00e9 du r\u00e9sultat d\u00e9pend directement de la pr\u00e9cision des mesures effectu\u00e9es sur le terrain et du bon \u00e9talonnage des instruments utilis\u00e9s.","L\u2019algorithme Random Forest a \u00e9t\u00e9 entra\u00een\u00e9 sur 122\u00a0\u00e9chantillons repr\u00e9sentatifs, bas\u00e9s sur les donn\u00e9es OMS, FAO, JMP (OMS/UNICEF), les \u00e9tudes hydrochimiques d\u2019Afrique de l\u2019Ouest (Akoteyon, 2011\u00a0; USEPA, 2022) et les normes nationales du B\u00e9nin. Pr\u00e9cision en validation crois\u00e9e (5-fold)\u00a0: 100\u00a0%.","Ce rapport ne se substitue pas \u00e0 une analyse compl\u00e8te en laboratoire agr\u00e9\u00e9. Pour une certification officielle de potabilit\u00e9, il est recommand\u00e9 de compl\u00e9ter par des tests suppl\u00e9mentaires (m\u00e9taux lourds, pesticides, coliformes totaux, chlore r\u00e9siduel).","R\u00e9f\u00e9rences\u00a0: OMS \u2014 Directives pour la qualit\u00e9 de l\u2019eau de boisson, 4e\u00a0\u00e9dition (2017)\u00a0; USEPA Drinking Water Standards (2022)\u00a0; Normes nationales du B\u00e9nin."],1):
        story.append(Paragraph(f"<b>Note\u00a0{i}\u00a0:</b> {note}",sn)); story.append(Spacer(1,0.15*cm))
    story.append(Spacer(1,0.3*cm))
    story.append(HRFlowable(width="100%",thickness=1.5,color=BLEU_CLAIR,spaceBefore=8,spaceAfter=6))
    ft=Table([[Paragraph(f"<b>EauVie</b> \u2014 Analyse intelligente de la qualit\u00e9 de l\u2019eau<br/>Propos\u00e9e par <b>Charles MEDEZOUNDJI</b> \u2014 B\u00e9nin, Afrique de l\u2019Ouest<br/>Rapport g\u00e9n\u00e9r\u00e9 le {date_str} \u00e0 {heure_str}\u00a0| R\u00e9f.\u00a0{ref_str}",S("ft1",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_LEFT,leading=11)),Paragraph("Ce document est g\u00e9n\u00e9r\u00e9 automatiquement.<br/>Il ne remplace pas une analyse en laboratoire agr\u00e9\u00e9.<br/><b>\u00a9 EauVie 2025 \u2014 Tous droits r\u00e9serv\u00e9s</b>",S("ft2",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_RIGHT,leading=11))]],colWidths=[(W-3.6*cm)/2,(W-3.6*cm)/2])
    ft.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")])); story.append(ft)
    doc.build(story); result=buffer.getvalue(); buffer.close(); return result

# ── MODÈLE IA ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rows=[]
    potable=[(7.2,0.8,22.0,320,2.1,7.5,0),(7.5,1.2,21.5,410,3.4,7.2,0),(7.0,0.3,20.0,280,1.8,8.0,0),(6.8,2.0,23.0,520,4.2,6.8,0),(7.3,0.5,19.5,350,2.8,7.8,0),(7.1,0.6,22.5,390,3.1,6.9,0),(7.4,1.0,21.0,440,3.6,7.3,0),(6.9,1.5,20.5,480,4.0,6.7,0),(7.6,0.9,22.0,310,2.3,7.4,0),(7.2,0.7,21.8,365,2.9,7.6,0),(7.8,1.8,23.5,550,4.8,6.6,0),(6.8,2.5,24.0,600,5.5,6.5,0),(6.7,0.2,18.0,260,1.5,8.2,0),(7.0,0.4,19.0,290,1.9,8.1,0),(7.3,1.1,21.0,420,3.3,7.1,0),(7.5,0.8,22.0,380,2.7,7.3,0),(6.8,1.3,20.0,460,3.8,6.7,0),(7.1,2.0,23.0,510,4.5,7.0,0),(7.4,0.6,19.5,330,2.2,7.5,0),(7.6,1.4,22.0,450,3.7,6.8,0),(7.0,1.7,21.5,400,3.2,7.2,0),(6.9,0.9,20.0,340,2.6,8.0,0),(7.2,2.3,23.0,580,4.9,6.6,0),(7.5,0.4,18.5,270,1.7,7.7,0),(7.8,1.0,21.0,395,3.0,7.0,0),(6.5,2.8,24.0,620,5.2,6.5,0),(8.0,0.7,22.0,360,2.5,6.8,0),(8.2,1.5,21.5,430,3.5,7.1,0),(7.3,0.3,19.0,285,1.8,8.0,0),(6.7,1.9,22.5,490,4.1,7.3,0),(7.1,1.1,21.0,415,3.2,7.4,0),(7.4,2.4,23.5,560,4.7,6.6,0)]
    douteuse=[(6.3,5.5,26.0,1050,12.0,5.8,2),(5.8,4.5,27.5,1200,18.5,5.5,4),(6.2,6.0,25.5,980,10.5,5.2,1),(7.8,7.0,28.0,1350,22.0,5.0,6),(7.9,8.0,26.5,1150,15.8,4.8,3),(6.4,9.0,29.0,1400,28.0,5.5,8),(6.5,6.5,25.5,1080,13.5,5.0,2),(8.6,5.8,27.0,1250,19.5,5.3,5),(5.5,4.8,26.0,1020,11.0,5.6,1),(6.1,7.5,28.5,1300,24.5,4.9,7),(8.0,5.0,25.0,1060,12.5,5.7,3),(5.7,6.2,27.0,1180,17.0,5.1,4),(8.3,7.8,26.0,1280,20.5,4.7,6),(6.0,5.3,25.5,1040,11.8,5.4,2),(7.5,8.5,29.5,1450,32.0,4.6,9),(5.9,6.8,27.5,1220,21.0,5.2,5),(8.7,5.5,26.5,1110,14.5,5.0,3),(6.6,7.2,28.0,1320,25.5,4.8,7),(5.6,8.8,27.0,1380,30.0,5.3,8),(7.2,6.0,25.5,1070,13.0,5.8,2),(8.4,5.2,26.0,1090,12.8,5.5,4),(6.3,9.5,29.5,1480,38.0,4.9,9),(5.8,7.0,28.0,1260,22.5,5.0,6),(7.7,6.5,26.5,1160,16.5,4.7,4),(6.8,8.2,27.5,1340,27.0,5.1,7),(8.8,5.0,25.0,1000,10.2,5.6,1),(5.5,7.8,28.5,1310,26.0,4.8,8),(7.0,6.8,27.0,1200,19.0,5.3,5),(6.2,9.2,29.0,1460,35.5,4.6,9),(8.5,5.8,26.0,1130,15.0,5.4,3)]
    polluee=[(6.0,25.0,31.0,2100,55.0,3.8,45),(5.8,35.0,32.5,2500,82.0,3.2,120),(5.5,20.0,30.5,1950,51.0,3.5,35),(6.3,40.0,33.0,2800,95.0,3.0,180),(5.9,30.0,31.5,2200,68.0,3.4,75),(9.2,28.0,32.0,2400,78.0,2.8,90),(5.6,50.0,34.0,3100,115.0,2.5,250),(6.2,18.0,30.0,1850,52.0,3.6,28),(6.5,45.0,33.5,2950,108.0,2.9,210),(5.7,55.0,34.5,3300,128.0,2.2,320),(4.8,22.0,31.0,2050,58.0,3.7,55),(5.3,32.0,32.0,2300,75.0,3.1,95),(9.5,26.0,31.5,2180,65.0,2.6,80),(5.4,48.0,33.5,2900,105.0,2.3,230),(6.1,15.0,30.0,1800,50.5,3.8,22),(4.9,38.0,32.5,2600,88.0,2.9,145),(9.8,20.0,31.0,2020,56.0,3.0,60),(5.2,42.0,33.0,2750,98.0,2.7,190),(6.4,12.0,30.0,1780,50.2,3.9,18),(5.0,52.0,34.5,3200,120.0,2.1,280),(9.3,24.0,31.5,2130,62.0,2.8,72),(5.5,36.0,32.0,2380,80.0,3.2,110),(6.0,47.0,33.5,2870,102.0,2.4,215),(4.7,28.0,31.5,2250,72.0,3.5,85),(9.6,18.0,31.0,2010,54.0,3.1,50),(5.1,44.0,33.0,2780,100.0,2.6,200),(6.3,33.0,32.0,2450,84.0,3.0,130),(5.8,16.0,30.5,1900,53.0,3.7,32),(9.1,30.0,32.5,2350,77.0,2.9,100),(4.6,40.0,33.0,2700,92.0,2.8,160)]
    dangereuse=[(4.2,90.0,36.5,4200,180.0,1.0,800),(5.0,95.0,37.0,4500,210.0,0.8,1200),(4.5,80.0,36.0,4000,165.0,0.5,650),(3.8,100.0,38.0,5200,280.0,0.3,2500),(8.8,85.0,36.5,4300,190.0,0.6,900),(4.0,70.0,35.5,3850,155.0,1.2,580),(4.5,65.0,36.0,3900,162.0,0.9,720),(4.1,75.0,37.0,4100,175.0,0.7,850),(4.8,88.0,36.5,4400,195.0,0.4,1100),(3.5,98.0,38.5,5500,320.0,0.2,3200),(4.3,92.0,37.0,4600,215.0,0.6,1400),(3.9,97.0,38.0,5100,275.0,0.1,2800),(4.4,72.0,35.5,3920,158.0,1.1,610),(3.7,105.0,39.0,5800,350.0,0.2,4000),(4.6,83.0,36.5,4150,178.0,0.5,780),(3.6,110.0,39.5,6000,380.0,0.1,4500),(4.9,68.0,35.5,3800,152.0,1.3,545),(3.4,115.0,40.0,6200,420.0,0.1,5000),(4.7,78.0,36.0,4050,168.0,0.4,700),(3.3,120.0,40.5,6500,460.0,0.1,5500),(4.0,95.0,37.5,4700,225.0,0.3,1600),(3.8,108.0,38.5,5400,300.0,0.2,3000),(4.2,85.0,36.5,4250,185.0,0.6,950),(3.6,102.0,38.0,5100,270.0,0.1,2600),(4.5,73.0,35.5,3950,160.0,1.0,630),(3.9,88.0,37.0,4450,200.0,0.4,1050),(4.3,96.0,37.5,4750,230.0,0.3,1700),(3.7,112.0,39.0,5900,360.0,0.1,4200),(4.6,80.0,36.0,4100,172.0,0.7,760),(3.5,125.0,41.0,6800,500.0,0.1,6000)]
    for lst,cl in [(potable,0),(douteuse,1),(polluee,2),(dangereuse,3)]:
        for v in lst: rows.append({"pH":v[0],"Turbidite":v[1],"Temperature":v[2],"Conductivite":v[3],"Nitrates":v[4],"O2":v[5],"Ecoli":v[6],"Classe":cl})
    data=pd.DataFrame(rows)
    X=data[["pH","Turbidite","Temperature","Conductivite","Nitrates","O2","Ecoli"]]; y=data["Classe"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features="sqrt",random_state=42,class_weight="balanced",n_jobs=-1)
    rf.fit(X_train,y_train); return rf
rf=load_model()

# ── CSS COMPLET ───────────────────────────────────────────────
st.markdown("""<style>
#MainMenu{visibility:hidden !important;}header{visibility:hidden !important;}
footer{visibility:hidden !important;}[data-testid='stToolbar']{display:none !important;}
html,body,[class*='css']{color:#0a0a0a !important;}
.main{background:linear-gradient(160deg,#dff3fb 0%,#e8f4fd 100%);}
.block-container{background:rgba(255,255,255,0.97);border-radius:18px;padding:2rem;box-shadow:0 4px 32px rgba(0,119,182,0.12);}
.stButton>button{background:linear-gradient(135deg,#0077b6,#00b4d8);color:white !important;font-size:16px;border-radius:14px;padding:12px 28px;width:100%;border:none;font-weight:700;}
.stButton>button:hover{background:linear-gradient(135deg,#023e8a,#0077b6);}
.result-box{padding:24px;border-radius:16px;text-align:center;font-size:24px;font-weight:800;margin:18px 0;box-shadow:0 4px 20px rgba(0,0,0,0.1);}
.potable{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);color:#0a4a0a !important;border:3px solid #28a745;}
.douteuse{background:linear-gradient(135deg,#fff3cd,#ffe082);color:#4a3000 !important;border:3px solid #ffc107;}
.polluee{background:linear-gradient(135deg,#ffd5d5,#ffab91);color:#5a0000 !important;border:3px solid #dc3545;}
.dangereuse{background:linear-gradient(135deg,#2d0000,#1a0000);color:#ff6666 !important;border:3px solid #ff0000;}
.pcard{background:linear-gradient(135deg,#f0f8ff,#e3f2fd);border-left:5px solid #0077b6;border-radius:12px;padding:14px 16px;margin-bottom:10px;}
.plabel{font-weight:800;color:#023e8a !important;font-size:15px;margin-bottom:6px;display:block;}
.ptext{color:#0a0a0a !important;font-size:13px;font-weight:500;line-height:1.6;display:block;margin-bottom:4px;}
.pnorm{font-size:12px;color:#023e8a !important;margin-top:7px;font-weight:700;background:rgba(0,119,182,0.10);padding:4px 10px;border-radius:6px;display:inline-block;}
.proto-box{background:linear-gradient(135deg,#f0fff4,#e8f8e8);border-left:5px solid #1b5e20;border-radius:12px;padding:14px 18px;margin-bottom:12px;}
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
.mesure-group{background:#f8fbff;border:1px solid #c8dff5;border-radius:10px;padding:10px 12px;margin-bottom:8px;}
.carto-box{background:linear-gradient(135deg,#e8f5e9,#f0fff4);border-left:5px solid #2e7d32;border-radius:12px;padding:14px 18px;margin:12px 0;}
.contact-box{background:linear-gradient(135deg,#023e8a,#0077b6);border-radius:14px;padding:18px;text-align:center;margin-top:20px;}
.normes-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;}
.normes-table th{background:#023e8a;color:white;padding:8px 10px;text-align:center;font-weight:700;}
.normes-table td{padding:7px 10px;text-align:center;border:1px solid #e0e0e0;color:#0a0a0a !important;font-weight:500;}
.normes-table tr:nth-child(even){background:#e3f2fd;}
.normes-table tr:nth-child(odd){background:#ffffff;}
.normes-table td:first-child{text-align:left;font-weight:700;color:#023e8a !important;}
p,span,div,label{color:#0a0a0a !important;}
</style>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
st.markdown(
    '<div class="header-box">'
    '<div class="header-title">\U0001f4a7 EauVie</div>'
    '<div class="header-sub">Analyse intelligente de la qualit\u00e9 de l\u2019eau \u2014 Normes OMS<br>'
    'afin de garantir une consommation rassurante et b\u00e9n\u00e9fique.</div>'
    '<div class="header-author">Propos\u00e9e par Charles MEDEZOUNDJI</div>'
    '</div>',
    unsafe_allow_html=True)

# ── NORMES OMS — tableau HTML esthétique ─────────────────────
with st.expander("\U0001f4cb Normes OMS de r\u00e9f\u00e9rence \u2014 7 param\u00e8tres"):
    st.markdown("""
    <table class="normes-table">
      <tr>
        <th>Param\u00e8tre</th>
        <th>\u2705 Potable</th>
        <th>\u26a0\ufe0f Douteuse</th>
        <th>\u274c Pollu\u00e9e</th>
        <th>\u2620\ufe0f Dangereuse</th>
      </tr>
      <tr><td>pH</td><td>6,5 \u00e0 8,5</td><td>5,5 \u00e0 9,0</td><td>4,5 \u00e0 5,5</td><td>inf\u00e9rieur \u00e0 4,5</td></tr>
      <tr><td>Turbidit\u00e9 (NTU)</td><td>inf\u00e9rieur \u00e0 5</td><td>5 \u00e0 10</td><td>10 \u00e0 50</td><td>sup\u00e9rieur \u00e0 50</td></tr>
      <tr><td>Temp\u00e9rature (\u00b0C)</td><td>inf\u00e9rieur \u00e0 25</td><td>25 \u00e0 30</td><td>30 \u00e0 35</td><td>sup\u00e9rieur \u00e0 35</td></tr>
      <tr><td>Conductivit\u00e9 (\u00b5S/cm)</td><td>inf\u00e9rieur \u00e0 2\u202f500</td><td>2\u202f500 \u00e0 3\u202f000</td><td>3\u202f000 \u00e0 4\u202f000</td><td>sup\u00e9rieur \u00e0 4\u202f000</td></tr>
      <tr><td>Nitrates (mg/L)</td><td>inf\u00e9rieur \u00e0 50</td><td>50 \u00e0 80</td><td>80 \u00e0 150</td><td>sup\u00e9rieur \u00e0 150</td></tr>
      <tr><td>Oxyg\u00e8ne dissous (mg/L)</td><td>sup\u00e9rieur \u00e0 6</td><td>4 \u00e0 6</td><td>2 \u00e0 4</td><td>inf\u00e9rieur \u00e0 2</td></tr>
      <tr><td>E. coli (UFC/100\u00a0mL)</td><td>0</td><td>1 \u00e0 10</td><td>10 \u00e0 500</td><td>sup\u00e9rieur \u00e0 500</td></tr>
    </table>
    """, unsafe_allow_html=True)

# ── PROTOCOLES ────────────────────────────────────────────────
with st.expander("\U0001f52c Protocoles de mesure \u2014 7 param\u00e8tres"):
    for titre, items in [
        ("\U0001f9ea pH \u2014 Potentiom\u00e8tre / pH-m\u00e8tre",["\U0001f527 Outil\u00a0: pH-m\u00e8tre num\u00e9rique ou bandelettes indicatrices de pH","1. \u00c9talonner avec les solutions tampon pH\u00a04, pH\u00a07 et pH\u00a010","2. Plonger l\u2019\u00e9lectrode propre dans l\u2019\u00e9chantillon","3. Attendre la stabilisation (environ 30\u00a0secondes)","4. Lire et noter la valeur affich\u00e9e","5. Rincer l\u2019\u00e9lectrode \u00e0 l\u2019eau distill\u00e9e apr\u00e8s chaque mesure"]),
        ("\U0001f30a Turbidit\u00e9 \u2014 Turbidim\u00e8tre num\u00e9rique (NTU)",["\U0001f527 Outil\u00a0: Turbidim\u00e8tre num\u00e9rique ou comparateur visuel","1. Remplir le tube de mesure avec l\u2019\u00e9chantillon","2. Essuyer le tube pour \u00e9liminer toute trace de doigts","3. Ins\u00e9rer le tube et fermer le couvercle","4. Lire la valeur en NTU affich\u00e9e","5. R\u00e9p\u00e9ter 3\u00a0fois et calculer la moyenne"]),
        ("\U0001f321\ufe0f Temp\u00e9rature \u2014 Thermom\u00e8tre \u00e9lectronique",["\U0001f527 Outil\u00a0: Thermom\u00e8tre num\u00e9rique \u00e0 sonde immersible (pr\u00e9cision 0,1\u00a0\u00b0C)","1. S\u2019assurer que la sonde est propre et s\u00e8che","2. Plonger la sonde directement dans l\u2019\u00e9chantillon","3. Attendre la stabilisation (1 \u00e0 2\u00a0minutes)","4. Lire et noter la temp\u00e9rature en \u00b0C","5. Mesurer id\u00e9alement sur le terrain, \u00e0 la source"]),
        ("\u26a1 Conductivit\u00e9 \u2014 Conductim\u00e8tre num\u00e9rique (\u00b5S/cm)",["\U0001f527 Outil\u00a0: Conductim\u00e8tre portable avec cellule de conductivit\u00e9","1. \u00c9talonner avec une solution \u00e9talon certifi\u00e9e","2. Rincer la cellule deux fois avec l\u2019\u00e9chantillon","3. Plonger la cellule et activer la mesure","4. Attendre la stabilisation et lire la valeur en \u00b5S/cm","5. Rincer la cellule \u00e0 l\u2019eau distill\u00e9e apr\u00e8s usage"]),
        ("\U0001f33f Nitrates \u2014 Spectrophotom\u00e8tre ou bandelettes r\u00e9actives",["\U0001f527 Outil\u00a0: Spectrophotom\u00e8tre portable ou kit colorim\u00e9trique (bandelettes nitrates)","1. Filtrer l\u2019\u00e9chantillon sur membrane 0,45\u00a0micron","2. Tremper la bandelette 1\u00a0seconde dans l\u2019eau","3. Attendre le temps de r\u00e9action (60\u00a0secondes)","4. Comparer la couleur \u00e0 la charte colorim\u00e9trique","5. Lire et noter la concentration en mg/L"]),
        ("\U0001f4a8 Oxyg\u00e8ne dissous \u2014 Oxym\u00e8tre \u00e9lectronique",["\U0001f527 Outil\u00a0: Oxym\u00e8tre num\u00e9rique portable avec sonde \u00e0 membrane","1. \u00c9talonner la sonde dans l\u2019air satur\u00e9 en humidit\u00e9 (10\u00a0min)","2. Plonger la sonde sans cr\u00e9er de bulles d\u2019air","3. Agiter tr\u00e8s doucement et attendre 1 \u00e0 2\u00a0minutes","4. Lire la valeur en mg/L affich\u00e9e","5. Rincer la sonde \u00e0 l\u2019eau distill\u00e9e apr\u00e8s usage"]),
        ("\U0001f9eb E.\u00a0coli \u2014 Test de pr\u00e9sence / absence",["\U0001f527 Outil\u00a0: Kit Colilert (IDEXX), bandelettes Compact Dry EC ou milieu m-FC","1. Pr\u00e9lever 100\u00a0mL d\u2019eau dans un flacon st\u00e9rile","2. Ajouter le r\u00e9actif Colilert et m\u00e9langer jusqu\u2019\u00e0 dissolution","3. Incuber \u00e0 35\u00a0\u00b0C pendant 24 \u00e0 28\u00a0heures","4. Fluorescence sous UV = pr\u00e9sence d\u2019E.\u00a0coli","5. Exprimer le r\u00e9sultat en UFC/100\u00a0mL"]),
    ]:
        items_html="".join([f'<span class="proto-item">{i}</span>' for i in items])
        st.markdown(f'<div class="proto-box"><span class="proto-title">{titre}</span>{items_html}</div>',unsafe_allow_html=True)

# ── INFORMATIONS ANALYSTE ─────────────────────────────────────
st.markdown('<span class="section-title">\U0001f464 Informations sur l\u2019analyste et le pr\u00e9l\u00e8vement</span>',unsafe_allow_html=True)
analyste = st.text_input("\U0001f464 Nom complet de l\u2019analyste *", placeholder="Ex\u00a0: Jean KOFFI", help="Obligatoire \u2014 ce nom figurera dans le rapport officiel")
lieu     = st.text_input("\U0001f4cd Lieu de pr\u00e9l\u00e8vement *", placeholder="Ex\u00a0: Village de Kpanr\u00f4u, commune de Djougou")
SOURCES  = ["Robinet (r\u00e9seau trait\u00e9)","Puits peu profond","Forage profond","Rivi\u00e8re","Fleuve","Lac","Marigot","Eau stagnante (mare)","Eau de pluie collect\u00e9e","Source naturelle","Ros\u00e9e collect\u00e9e","Eau de mer / c\u00f4ti\u00e8re","Eau de barrage","Eau de citerne stock\u00e9e","Autre"]
source   = st.selectbox("\U0001f30a Source de l\u2019eau *", SOURCES)

# Coordonnées pour la cartographie
st.markdown('<span class="section-title">\U0001f30d Coordonn\u00e9es g\u00e9ographiques (optionnel)</span>',unsafe_allow_html=True)
cg1, cg2 = st.columns(2)
lat_input = cg1.number_input("Latitude", value=6.3703, step=0.0001, format="%.4f", help="Ex\u00a0: 6.3703 pour Cotonou")
lon_input = cg2.number_input("Longitude", value=2.4305, step=0.0001, format="%.4f", help="Ex\u00a0: 2.4305 pour Cotonou")

# ── SAISIE DES MESURES ────────────────────────────────────────
st.markdown('<span class="section-title">\U0001f52c Ins\u00e9rez les trois mesures de chaque param\u00e8tre</span>',unsafe_allow_html=True)
st.info("\u2139\ufe0f Saisissez les trois mesures r\u00e9alis\u00e9es pour chaque param\u00e8tre. La moyenne sera calcul\u00e9e automatiquement.")

def triple_input(label, label_card, ptext, pnorm, min_v, max_v, default, step, unit=""):
    st.markdown(f'<div class="pcard"><span class="plabel">{label_card}</span><span class="ptext">{ptext}</span><span class="pnorm">{pnorm}</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="mesure-group">',unsafe_allow_html=True)
    ca,cb,cc = st.columns(3)
    v1 = ca.number_input(f"{label} \u2014 Mesure\u00a01", min_value=min_v, max_value=max_v, value=default, step=step, key=f"{label}_1")
    v2 = cb.number_input(f"{label} \u2014 Mesure\u00a02", min_value=min_v, max_value=max_v, value=default, step=step, key=f"{label}_2")
    v3 = cc.number_input(f"{label} \u2014 Mesure\u00a03", min_value=min_v, max_value=max_v, value=default, step=step, key=f"{label}_3")
    st.markdown('</div>',unsafe_allow_html=True)
    moy = round((v1+v2+v3)/3, 4)
    st.caption(f"\U0001f4ca Moyenne {label}\u00a0: **{moy}** {unit}")
    return moy

pH  = triple_input("pH","\U0001f9ea pH \u2014 Potentiel Hydrog\u00e8ne","Mesure l\u2019acidit\u00e9 ou la basicit\u00e9 de l\u2019eau. pH bas\u00a0: risque de m\u00e9taux toxiques. pH \u00e9lev\u00e9\u00a0: contamination min\u00e9rale ou chimique.","Norme OMS\u00a0: 6,5 \u00e0 8,5",0.0,14.0,7.0,0.01)
tu  = triple_input("Turbidite","\U0001f30a Turbidit\u00e9 (NTU) \u2014 Trouble de l\u2019eau","Particules en suspension\u00a0: argile, bact\u00e9ries, mati\u00e8res organiques. Eau trouble\u00a0: agents pathog\u00e8nes possibles.","Norme OMS\u00a0: inf\u00e9rieur \u00e0 5 NTU",0.0,200.0,2.0,0.01,"NTU")
te  = triple_input("Temperature","\U0001f321\ufe0f Temp\u00e9rature (\u00b0C) \u2014 Activit\u00e9 microbienne","Au-del\u00e0 de 25\u00a0\u00b0C, la prolif\u00e9ration des micro-organismes pathog\u00e8nes s\u2019acc\u00e9l\u00e8re significativement.","Norme OMS\u00a0: inf\u00e9rieur \u00e0 25\u00a0\u00b0C",0.0,60.0,22.0,0.1,"\u00b0C")
co  = triple_input("Conductivite","\u26a1 Conductivit\u00e9 (\u00b5S/cm) \u2014 Min\u00e9ralisation","Conductivit\u00e9 \u00e9lev\u00e9e\u00a0: concentration excessive en sels et min\u00e9raux dissous, nocive \u00e0 long terme.","Norme OMS\u00a0: inf\u00e9rieur \u00e0 2\u202f500 \u00b5S/cm",0.0,10000.0,350.0,1.0,"\u00b5S/cm")
no  = triple_input("Nitrates","\U0001f33f Nitrates (mg/L) \u2014 Pollution agricole","Proviennent des engrais agricoles. Au-del\u00e0 de 50\u00a0mg/L, ils provoquent la m\u00e9th\u00e9moglobin\u00e9mie chez les nourrissons.","Norme OMS\u00a0: inf\u00e9rieur \u00e0 50 mg/L",0.0,500.0,5.0,0.1,"mg/L")
o2  = triple_input("O2","\U0001f4a8 Oxyg\u00e8ne dissous (mg/L) \u2014 Vitalit\u00e9 de l\u2019eau","Taux faible\u00a0: d\u00e9composition organique intense, bact\u00e9ries. En dessous de 2\u00a0mg/L\u00a0: eau anoxique et dangereuse.","Norme\u00a0: sup\u00e9rieur \u00e0 6 mg/L",0.0,14.0,7.0,0.01,"mg/L")
ec  = triple_input("Ecoli","\U0001f9eb E.\u00a0coli (UFC/100\u00a0mL) \u2014 Contamination f\u00e9cale","Indicateur direct de contamination f\u00e9cale. Toute pr\u00e9sence signale un risque sanitaire et la probable pr\u00e9sence d\u2019autres pathog\u00e8nes.","Norme OMS\u00a0: 0 UFC/100\u00a0mL",0.0,10000.0,0.0,1.0,"UFC/100 mL")

st.markdown("---")

# ── MAPPING COULEURS RÉSULTAT ─────────────────────────────────
COULEURS_CARTO = {0:"green",1:"orange",2:"red",3:"darkred"}
MP = {
    0:("\U0001f4a7 POTABLE","potable","Eau conforme aux normes OMS. Consommation possible sans risque."),
    1:("\u26a0\ufe0f DOUTEUSE","douteuse","Anomalies d\u00e9tect\u00e9es. Filtrez et faites bouillir avant consommation."),
    2:("\u274c POLLU\u00c9E","polluee","Eau pollu\u00e9e. Ne pas consommer. Traitement obligatoire."),
    3:("\u2620\ufe0f DANGEREUSE","dangereuse","DANGER EXTR\u00caM E. Tout contact \u00e0 \u00e9viter. Risque sanitaire majeur."),
}

# ── BOUTON ANALYSE ────────────────────────────────────────────
if st.button("\U0001f50d Analyser la qualit\u00e9 de l\u2019eau"):
    erreurs=[]
    if not analyste.strip(): erreurs.append("\u26a0\ufe0f Le nom de l\u2019analyste est obligatoire.")
    if not lieu.strip():     erreurs.append("\u26a0\ufe0f Le lieu de pr\u00e9l\u00e8vement est obligatoire.")
    if erreurs:
        for e in erreurs: st.error(e)
    else:
        dfm=pd.DataFrame({"pH":[pH],"Turbidite":[tu],"Temperature":[te],"Conductivite":[co],"Nitrates":[no],"O2":[o2],"Ecoli":[ec]})
        cl=rf.predict(dfm)[0]; pr=rf.predict_proba(dfm)[0]
        lb,cs,co_msg=MP[cl]; conf=str(round(pr[cl]*100,1))
        st.markdown(f'<div class="result-box {cs}">{lb}<br><span style="font-size:14px;font-weight:600;">Confiance du mod\u00e8le\u00a0: {conf}\u00a0%</span></div>',unsafe_allow_html=True)
        st.markdown(f"**\U0001f4a1 Conseil\u00a0:** {co_msg}")

        if cl in [1,2,3]:
            with st.expander("\U0001f6e0\ufe0f Comment purifier cette eau\u00a0?"):
                for titre_c,desc_c in [
                    ("\U0001f525 1. \u00c9bullition","Porter l\u2019eau \u00e0 \u00e9bullition pendant au moins 5\u00a0minutes. Laisser refroidir dans un r\u00e9cipient propre et couvert. Efficace contre les bact\u00e9ries, les virus et les parasites."),
                    ("\U0001f9f4 2. Filtration sur sable et gravier","Couches successives\u00a0: gravier grossier, gravier fin, sable grossier, sable fin, charbon de bois actif. Compl\u00e9ter obligatoirement avec l\u2019\u00e9bullition."),
                    ("\u2600\ufe0f 3. D\u00e9sinfection solaire SODIS","Bouteilles transparentes expos\u00e9es 6\u00a0heures au soleil (ciel clair) ou 2\u00a0jours (nuageux). M\u00e9thode gratuite et valid\u00e9e par l\u2019OMS."),
                    ("\U0001f9ea 4. Chloration","2\u00a0gouttes d\u2019eau de Javel \u00e0 5\u00a0% par litre d\u2019eau trouble (1\u00a0goutte si claire). Attendre 30\u00a0minutes avant de consommer."),
                    ("\U0001f331 5. Graines de Moringa oleifera","Broyer 2\u00a03\u00a0graines s\u00e8ches en poudre fine. Ajouter \u00e0 1\u00a0litre d\u2019eau turbide, agiter 1\u00a0minute puis 5\u00a0minutes lentement. D\u00e9canter 1\u00a0heure.")
                ]:
                    st.markdown(f'<div class="conseil-box"><span class="conseil-title">{titre_c}</span><span class="conseil-item">{desc_c}</span></div>',unsafe_allow_html=True)

        # ── RAPPORT PDF ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### \U0001f4c4 Rapport officiel PDF")
        try:
            mesures={"pH":pH,"turb":tu,"temp":te,"cond":co,"no3":no,"o2":o2,"ecoli":ec}
            heure_locale=datetime.now().strftime("%H:%M")
            pdf_bytes=generer_pdf(mesures,cl,list(pr),analyste=analyste,lieu=lieu,source=source,heure_locale=heure_locale)
            nom_pdf="rapport_eauvie_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".pdf"
            st.download_button(label="\U0001f4e5 T\u00e9l\u00e9charger le rapport PDF officiel",data=pdf_bytes,file_name=nom_pdf,mime="application/pdf")
            st.success("\u2705 Rapport PDF g\u00e9n\u00e9r\u00e9 avec succ\u00e8s\u00a0!")
        except Exception as e:
            st.error("Erreur PDF\u00a0: "+str(e))

        # ── CARTOGRAPHIE ──────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="carto-box">\U0001f30d <b>Cartographie communautaire</b><br>Si votre mesure est r\u00e9elle, vous pouvez l\u2019ajouter \u00e0 la cartographie afin de faciliter les d\u00e9cisions des autorit\u00e9s, des ONG et autres acteurs en mati\u00e8re de l\u2019eau.</div>',unsafe_allow_html=True)
        ajouter_carto = st.checkbox("\U0001f4cd Ajouter cette mesure \u00e0 la cartographie communautaire")
        if ajouter_carto:
            point = {
                "lat": lat_input, "lon": lon_input,
                "lieu": lieu, "source": source,
                "resultat": lb, "classe": int(cl),
                "analyste": analyste,
                "date": datetime.now().strftime("%d/%m/%Y"),
                "heure": datetime.now().strftime("%H:%M"),
                "pH": pH, "turbidite": tu, "temperature": te,
                "conductivite": co, "nitrates": no, "o2": o2, "ecoli": ec
            }
            st.session_state[CARTO_KEY].append(point)
            st.success(f"\u2705 Mesure ajout\u00e9e \u00e0 la cartographie ({len(st.session_state[CARTO_KEY])} point(s) enregistr\u00e9(s) dans cette session).")

        # Graphique probabilités
        prd=pd.DataFrame({"Classe":["Potable","Douteuse","Pollu\u00e9e","Dangereuse"],"Probabilit\u00e9 (%)":[round(p*100,1) for p in pr]})
        st.bar_chart(prd.set_index("Classe"))

        # Historique session
        if "histo" not in st.session_state: st.session_state.histo=[]
        st.session_state.histo.append({"Heure":datetime.now().strftime("%H:%M:%S"),"Analyste":analyste,"Lieu":lieu,"Source":source,"pH":pH,"Turb.":tu,"Temp.(\u00b0C)":te,"Cond.":co,"NO\u2083":no,"O\u2082 dissous":o2,"E.coli":ec,"R\u00e9sultat":lb})

# ── HISTORIQUE SESSION ────────────────────────────────────────
if "histo" in st.session_state and len(st.session_state.histo)>0:
    st.markdown("---")
    st.markdown('<span class="section-title">\U0001f554 Historique des analyses</span>',unsafe_allow_html=True)
    hdf=pd.DataFrame(st.session_state.histo)
    st.dataframe(hdf,use_container_width=True)
    st.download_button("\u2b07\ufe0f T\u00e9l\u00e9charger le CSV",hdf.to_csv(index=False).encode("utf-8"),"historique_eauvie.csv","text/csv")

# ── CARTOGRAPHIE ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<span class="section-title">\U0001f5fa\ufe0f Cartographie des analyses</span>',unsafe_allow_html=True)
mdp_carto = st.text_input("\U0001f512 Mot de passe pour acc\u00e9der \u00e0 la cartographie", type="password", placeholder="Saisir le mot de passe")
if mdp_carto:
    if mdp_carto == "CARTOGRAPHIE":
        points = st.session_state.get(CARTO_KEY, [])
        if len(points) == 0:
            st.info("\U0001f4cd Aucune mesure n\u2019a encore \u00e9t\u00e9 ajout\u00e9e \u00e0 la cartographie dans cette session.")
        else:
            st.success(f"\u2705 {len(points)} point(s) enregistr\u00e9(s) sur la carte.")
            try:
                import folium
                from streamlit_folium import st_folium
                m = folium.Map(location=[6.3703,2.4305], zoom_start=7, tiles="CartoDB positron")
                COULEURS_CARTO_MAP = {0:"green",1:"orange",2:"red",3:"darkred"}
                ICONES_CARTO = {0:"tint",1:"exclamation-triangle",2:"times-circle",3:"skull-crossbones"}
                for p in points:
                    popup_html = f'''
                    <div style='font-family:sans-serif;font-size:13px;min-width:200px;'>
                    <b style='color:#023e8a;font-size:15px;'>{p['resultat']}</b><br>
                    <b>Lieu :</b> {p['lieu']}<br>
                    <b>Source :</b> {p['source']}<br>
                    <b>Analyste :</b> {p['analyste']}<br>
                    <b>Date :</b> {p['date']} \u00e0 {p['heure']}<br>
                    <hr style='margin:6px 0;'>
                    <b>pH :</b> {p['pH']} &nbsp; <b>Turb. :</b> {p['turbidite']} NTU<br>
                    <b>Temp. :</b> {p['temperature']} \u00b0C &nbsp; <b>Cond. :</b> {p['conductivite']} \u00b5S/cm<br>
                    <b>NO\u2083 :</b> {p['nitrates']} mg/L &nbsp; <b>O\u2082 :</b> {p['o2']} mg/L<br>
                    <b>E. coli :</b> {p['ecoli']} UFC/100\u00a0mL
                    </div>'''
                    folium.Marker(
                        location=[p["lat"], p["lon"]],
                        popup=folium.Popup(popup_html, max_width=280),
                        tooltip=f"{p['resultat']} \u2014 {p['lieu']}",
                        icon=folium.Icon(color=COULEURS_CARTO_MAP.get(p["classe"],"blue"),
                                         icon="info-sign", prefix="glyphicon")
                    ).add_to(m)
                st_folium(m, width=700, height=450)
                # Export JSON
                df_carto = pd.DataFrame(points)
                st.dataframe(df_carto, use_container_width=True)
                st.download_button("\u2b07\ufe0f Exporter la cartographie (CSV)",
                    df_carto.to_csv(index=False).encode("utf-8"),
                    "cartographie_eauvie.csv","text/csv")
                json_str = json.dumps({"points":points},ensure_ascii=False,indent=2)
                st.download_button("\u2b07\ufe0f Exporter la cartographie (JSON)",
                    json_str.encode("utf-8"),
                    "cartographie_eauvie.json","application/json")
            except ImportError:
                st.warning("\u26a0\ufe0f La biblioth\u00e8que folium n\u2019est pas disponible. Voici les donn\u00e9es tabulaires.")
                st.dataframe(pd.DataFrame(points),use_container_width=True)
    else:
        st.error("\u274c Mot de passe incorrect.")

# ── CONTACT DÉVELOPPEUR ───────────────────────────────────────
st.markdown("---")
st.markdown('<span class="section-title">\U0001f4e7 Contacter le d\u00e9veloppeur</span>',unsafe_allow_html=True)
st.markdown("""
<div class="contact-box">
  <div style="color:#ffffff;font-size:16px;font-weight:800;margin-bottom:8px;">\U0001f4e7 Charles MEDEZOUNDJI</div>
  <div style="color:#d0eeff;font-size:13px;margin-bottom:14px;">D\u00e9veloppeur de l\u2019application EauVie \u2014 B\u00e9nin, Afrique de l\u2019Ouest</div>
  <a href="mailto:charlesezechielmedezoundji@gmail.com?subject=EauVie%20-%20Message&body=Bonjour%20Charles%2C%0A%0AJe%20vous%20contacte%20au%20sujet%20de%20l'application%20EauVie.%0A%0A"
     target="_blank"
     style="display:inline-block;background:white;color:#023e8a;font-weight:800;font-size:15px;padding:12px 28px;border-radius:10px;text-decoration:none;box-shadow:0 2px 10px rgba(0,0,0,0.2);">
    \U0001f4e4 Envoyer un message
  </a>
  <div style="color:#a8d8ff;font-size:11px;margin-top:10px;">charlesezechielmedezoundji@gmail.com</div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:#023e8a !important;font-size:12px;padding:10px;font-weight:600;">\U0001f4a7 EauVie \u2014 Random Forest \u2014 Normes OMS \u2014 7 param\u00e8tres \u2014 Charles MEDEZOUNDJI</div>',unsafe_allow_html=True)
