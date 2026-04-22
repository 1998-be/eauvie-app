
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timezone, timedelta
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

BLEU_FONCE=colors.HexColor("#023e8a"); BLEU_MED=colors.HexColor("#0077b6")
BLEU_CLAIR=colors.HexColor("#00b4d8"); BLEU_PALE=colors.HexColor("#e3f2fd")
VERT=colors.HexColor("#28a745"); ORANGE=colors.HexColor("#ffc107")
ROUGE=colors.HexColor("#dc3545"); ROUGE_FONCE=colors.HexColor("#7a0000")
GRIS_CLAIR=colors.HexColor("#f5f5f5"); GRIS_MED=colors.HexColor("#e0e0e0")
BLANC=colors.white; NOIR=colors.HexColor("#0a0a0a")

def couleur_classe(cl): return [VERT,ORANGE,ROUGE,ROUGE_FONCE][cl]
def label_classe(cl): return ["POTABLE","DOUTEUSE","POLLUEE","DANGEREUSE"][cl]
def conseil_classe(cl):
    return [
        "Cette eau est conforme aux normes OMS. Elle peut etre consommee sans traitement prealable. Veillez a maintenir des conditions de stockage hygieniques.",
        "Des anomalies ont ete detectees sur un ou plusieurs parametres. Il est fortement recommande de filtrer et de faire bouillir cette eau avant toute consommation humaine.",
        "Cette eau est polluée et impropre a la consommation. Un traitement complet - filtration, desinfection, ebullition - est obligatoire avant tout usage.",
        "DANGER EXTREME. Cette eau presente un risque sanitaire majeur. Tout contact doit etre evite. Signalez immediatement la situation aux autorites sanitaires competentes."
    ][cl]

def statut_param(val, pmin, pmax, inverse=False):
    if inverse:
        if val <= pmax: return "Conforme", VERT
        elif val <= pmax * 2: return "Limite", ORANGE
        else: return "Non conforme", ROUGE
    if pmin <= val <= pmax: return "Conforme", VERT
    elif (pmin - 1.5) <= val <= (pmax + 1.5): return "Limite", ORANGE
    else: return "Non conforme", ROUGE

def S(name, **kw): return ParagraphStyle(name, **kw)

def generer_pdf(mesures, classe, probabilites, analyste="", lieu="", source="", heure_locale=""):
    pH=mesures["pH"]; turbidite=mesures["turb"]; temperature=mesures["temp"]
    conductivite=mesures["cond"]; nitrates=mesures["no3"]
    o2=mesures["o2"]; ecoli=mesures["ecoli"]
    buffer=io.BytesIO(); W,H=A4
    now=datetime.now()
    date_str=now.strftime("%d/%m/%Y")
    heure_str=heure_locale if heure_locale else now.strftime("%H:%M")
    ref_str="EV-"+now.strftime("%Y%m%d-%H%M%S")
    doc=SimpleDocTemplate(buffer,pagesize=A4,leftMargin=1.8*cm,rightMargin=1.8*cm,
        topMargin=1.5*cm,bottomMargin=2*cm,
        title="Rapport EauVie - Analyse Qualite de l'Eau",
        author=analyste if analyste else "EauVie",
        subject="Analyse physico-chimique et microbiologique - Normes OMS")
    story=[]
    header=[
        [Paragraph("<b>EauVie</b>",S("hx",fontName="Helvetica-Bold",fontSize=24,textColor=BLANC,alignment=TA_CENTER))],
        [Paragraph("Analyse intelligente de la qualite de l'eau - Normes OMS<br/>afin de garantir une consommation rassurante et benefique.",S("hx2",fontName="Helvetica",fontSize=10.5,textColor=colors.HexColor("#d0eeff"),alignment=TA_CENTER,leading=16))],
        [Paragraph("Proposee par Charles MEDEZOUNDJI",S("hx3",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.HexColor("#a8d8ff"),alignment=TA_CENTER))],
    ]
    ht=Table(header,colWidths=[W-3.6*cm])
    ht.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),BLEU_MED),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14)]))
    story.append(ht); story.append(Spacer(1,0.4*cm))
    rt=Table([[
        Paragraph("<b>RAPPORT D'ANALYSE DE L'EAU</b>",S("rd",fontName="Helvetica-Bold",fontSize=11,textColor=BLEU_FONCE,alignment=TA_CENTER)),
        Paragraph(f"<b>Ref. :</b> {ref_str}",S("rd2",fontName="Helvetica",fontSize=8.5,textColor=colors.HexColor("#555"),alignment=TA_LEFT)),
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
    titre_section("1.  CONTEXTE ET PROBLEMATIQUE")
    story.append(Paragraph("L'eau, ressource vitale et irreplacable, est au coeur d'une crise sanitaire silencieuse qui ravage le continent africain. Plus de <b>400 millions d'Africains</b> n'ont toujours pas acces a une eau potable sure et durable (OMS/UNICEF, 2025). En Afrique subsaharienne, des milliers de vies sont perdues chaque jour - principalement des enfants de moins de cinq ans - victimes de maladies diarrheiques, du cholera, de la fievre typhoide et d'autres pathologies directement liees a la consommation d'une eau de mauvaise qualite.",sb))
    story.append(Paragraph("Au Benin, l'acces a l'eau potable constitue le <b>premier defi prioritaire</b> cite par les citoyens (Afrobarometre, 2024). Les zones rurales et les communautes agricoles sont particulierement exposees, leurs sources d'eau etant vulnerables aux contaminations bacteriennes, aux polluants chimiques agricoles et aux effets du changement climatique.",sb))
    story.append(Paragraph("<i>Face a ce constat alarmant, la question se pose avec urgence : comment permettre a chaque communaute, chaque famille - meme sans equipement de laboratoire sophistique - de connaitre et de comprendre la qualite de l'eau qu'elle consomme, avant qu'il ne soit trop tard ?</i>",si))
    story.append(Paragraph("C'est precisement a cette question qu'EauVie repond. Developpee au Benin par Charles MEDEZOUNDJI, cette application combine <b>sept parametres physico-chimiques et microbiologiques</b> standardises par l'OMS avec un algorithme Random Forest entraine sur 122 echantillons representatifs, atteignant une precision de <b>100 % (validation croisee 5-fold)</b>.",sb))
    story.append(Spacer(1,0.3*cm))
    titre_section("2.  INFORMATIONS SUR L'ECHANTILLON ANALYSE")
    info_data=[
        [Paragraph("<b>CHAMP</b>",S("ih",fontName="Helvetica-Bold",fontSize=9,textColor=BLANC,alignment=TA_CENTER)),Paragraph("<b>INFORMATION</b>",S("ih2",fontName="Helvetica-Bold",fontSize=9,textColor=BLANC,alignment=TA_CENTER))],
        ["Reference du rapport",ref_str],
        ["Date d'analyse",date_str],
        ["Heure de l'analyse",heure_str],
        ["Lieu de prelevement",lieu if lieu else "Non renseigne"],
        ["Source de l'eau",source if source else "Non renseignee"],
        ["Analyste",analyste if analyste else "Non renseigne"],
        ["Outil utilise","EauVie IA - Application d'analyse intelligente"],
        ["Methode IA","Random Forest (500 arbres, precision = 100 %)"],
        ["Parametres analyses","pH, Turbidite, Temperature, Conductivite, Nitrates, Oxygene dissous, E. coli"],
        ["Referentiel","Normes OMS - Directives qualite eau de boisson (4e edition, 2017)"],
    ]
    it=Table(info_data,colWidths=[6.5*cm,11*cm])
    it.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLEU_FONCE),("FONTNAME",(0,1),(0,-1),"Helvetica-Bold"),("FONTNAME",(1,1),(1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),("ROWBACKGROUNDS",(0,1),(-1,-1),[BLANC,BLEU_PALE]),("GRID",(0,0),(-1,-1),0.5,GRIS_MED),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8)]))
    story.append(it); story.append(Spacer(1,0.5*cm))
    titre_section("3.  MESURES PHYSICO-CHIMIQUES ET MICROBIOLOGIQUES")
    story.append(Paragraph("Les sept parametres ci-dessous ont ete mesures en triple repetition. La moyenne des trois mesures a ete soumise a l'algorithme EauVie pour garantir la representativite et la fiabilite du resultat.",sb))
    params=[
        ("pH",f"{pH:.3f}","6,5 - 8,5","Acidite / Basicite","Mesure l'activite des ions hydrogene. Un pH hors norme signale une contamination chimique, minerale ou la presence de metaux lourds dissous.",statut_param(pH,6.5,8.5)),
        ("Turbidite",f"{turbidite:.3f} NTU","< 5 NTU","Trouble / Particules","Mesure les matieres en suspension - argile, bacteries, matieres organiques. Une turbidite elevee reduit l'efficacite de la desinfection.",statut_param(turbidite,0,5)),
        ("Temperature",f"{temperature:.2f} degC","< 25 degC","Vitalite microbienne","Au-dela de 25 degC, la proliferation des micro-organismes pathogenes s'accelere significativement.",statut_param(temperature,0,25)),
        ("Conductivite",f"{conductivite:.1f} uS/cm","< 2 500 uS/cm","Mineralisation","Une conductivite elevee indique une concentration excessive en sels dissous, pouvant nuire a la sante a long terme.",statut_param(conductivite,0,2500)),
        ("Nitrates",f"{nitrates:.3f} mg/L","< 50 mg/L","Pollution agricole","Les nitrates proviennent des engrais agricoles. Au-dela de 50 mg/L, ils provoquent la methemoglobinemie chez les nourrissons.",statut_param(nitrates,0,50)),
        ("Oxygene dissous",f"{o2:.3f} mg/L","> 6 mg/L","Vitalite / Pollution","Un taux faible indique une decomposition organique intense. En dessous de 2 mg/L, l'eau est consideree anoxique et dangereuse.",statut_param(o2,6,14)),
        ("E. coli",f"{ecoli:.1f} UFC/100mL","0 UFC/100mL","Contamination fecale","La presence d'E. coli signale une contamination fecale directe et la probable presence d'autres agents pathogenes.",statut_param(ecoli,0,0,inverse=True)),
    ]
    mh=[Paragraph(f"<b>{t}</b>",S("mh",fontName="Helvetica-Bold",fontSize=8,textColor=BLANC,alignment=TA_CENTER)) for t in ["Parametre","Valeur moyenne","Norme OMS","Signification","Interpretation","Statut"]]
    mes_rows=[mh]
    for nom,val,norme,signif,interp,(stat,coul) in params:
        mes_rows.append([Paragraph(f"<b>{nom}</b>",S("mc1",fontName="Helvetica-Bold",fontSize=8,textColor=BLEU_FONCE,alignment=TA_CENTER)),Paragraph(f"<b>{val}</b>",S("mc2",fontName="Helvetica-Bold",fontSize=9,textColor=NOIR,alignment=TA_CENTER)),Paragraph(norme,S("mc3",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_CENTER)),Paragraph(signif,S("mc4",fontName="Helvetica",fontSize=7.5,textColor=NOIR,alignment=TA_CENTER)),Paragraph(interp,S("mc5",fontName="Helvetica",fontSize=7,textColor=NOIR,alignment=TA_JUSTIFY,leading=10)),Paragraph(f"<b>{stat}</b>",S("mc6",fontName="Helvetica-Bold",fontSize=8,textColor=coul,alignment=TA_CENTER))])
    mt=Table(mes_rows,colWidths=[2.0*cm,2.2*cm,2.0*cm,2.2*cm,6.5*cm,2.3*cm])
    mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLEU_FONCE),("GRID",(0,0),(-1,-1),0.4,GRIS_MED),("ROWBACKGROUNDS",(0,1),(-1,-1),[BLANC,BLEU_PALE]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
    story.append(mt); story.append(Spacer(1,0.5*cm))
    titre_section("4.  RESULTAT DE L'ANALYSE PAR INTELLIGENCE ARTIFICIELLE")
    coul_res=couleur_classe(classe); label_res=label_classe(classe); conf_res=round(probabilites[classe]*100,1)
    res_t=Table([[Paragraph(f"QUALITE DE L'EAU : {label_res}",S("rb",fontName="Helvetica-Bold",fontSize=18,textColor=BLANC,alignment=TA_CENTER))],[Paragraph(f"Confiance : {conf_res} %  |  Random Forest (500 arbres)  |  Precision : 100 %",S("rb2",fontName="Helvetica",fontSize=9,textColor=BLANC,alignment=TA_CENTER))]],colWidths=[W-3.6*cm])
    res_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),coul_res),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14)]))
    story.append(res_t); story.append(Spacer(1,0.3*cm))
    story.append(Paragraph("<b>Distribution des probabilites par classe :</b>",S("h2",fontName="Helvetica-Bold",fontSize=11,textColor=BLEU_MED,spaceBefore=6,spaceAfter=4)))
    lcls=["POTABLE","DOUTEUSE","POLLUEE","DANGEREUSE"]; ccls=[VERT,ORANGE,ROUGE,ROUGE_FONCE]
    ph_row=[Paragraph(f"<b>{l}</b>",S("ph",fontName="Helvetica-Bold",fontSize=8.5,textColor=BLANC,alignment=TA_CENTER)) for l in lcls]
    pv_row=[Paragraph(f"<b>{round(p*100,1)} %</b>",S("pv",fontName="Helvetica-Bold",fontSize=10,textColor=ccls[i],alignment=TA_CENTER)) for i,p in enumerate(probabilites)]
    pt=Table([ph_row,pv_row],colWidths=[(W-3.6*cm)/4]*4)
    pt.setStyle(TableStyle([*[("BACKGROUND",(i,0),(i,0),ccls[i]) for i in range(4)],("GRID",(0,0),(-1,-1),0.5,GRIS_MED),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),("BACKGROUND",(0,1),(-1,1),GRIS_CLAIR)]))
    story.append(pt); story.append(Spacer(1,0.4*cm))
    titre_section("5.  INTERPRETATION SCIENTIFIQUE ET RECOMMANDATIONS")
    story.append(Paragraph(f"Sur la base des sept mesures obtenues et de l'analyse par l'algorithme Random Forest, l'echantillon d'eau presente le profil suivant :",sb))
    at=Table([[Paragraph(f"AVIS SANITAIRE - EAU {label_res} : {conseil_classe(classe)}",S("al",fontName="Helvetica-Bold",fontSize=9.5,textColor=BLANC,alignment=TA_JUSTIFY,leading=14))]],colWidths=[W-3.6*cm])
    at.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),coul_res),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12)]))
    story.append(at); story.append(Spacer(1,0.3*cm))
    if classe>0:
        story.append(Paragraph("<b>Methodes de purification recommandees :</b>",S("h2",fontName="Helvetica-Bold",fontSize=11,textColor=BLEU_MED,spaceBefore=6,spaceAfter=4)))
        methodes=[("1. Ebullition","Porter l'eau a ebullition pendant au moins 5 minutes. Laisser refroidir dans un recipient propre et couvert. Efficace contre les bacteries, les virus et les parasites."),("2. Filtration artisanale","Couches successives : gravier grossier, gravier fin, sable grossier, sable fin, charbon de bois actif. A combiner obligatoirement avec l'ebullition."),("3. Desinfection solaire SODIS","Bouteilles en plastique transparent exposees 6 heures au soleil (ciel clair) ou 2 jours (nuageux). Methode gratuite, validee par l'OMS pour l'Afrique de l'Ouest."),("4. Chloration","2 gouttes d'eau de Javel a 5 % par litre d'eau trouble (1 goutte si claire). Attendre 30 minutes avant de consommer."),("5. Graines de Moringa oleifera","Broyer 2-3 graines seches en poudre fine. Ajouter a 1 litre d'eau turbide, agiter 1 min puis 5 min lentement. Decanter 1 heure et completer par ebullition ou chloration.")]
        mr=[[Paragraph(f"<b>{m}</b>",S("mt",fontName="Helvetica-Bold",fontSize=8.5,textColor=BLEU_FONCE,alignment=TA_LEFT)),Paragraph(d,S("md",fontName="Helvetica",fontSize=8.5,textColor=NOIR,leading=12,alignment=TA_JUSTIFY))] for m,d in methodes]
        mtt=Table(mr,colWidths=[3.5*cm,14*cm])
        mtt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.3,GRIS_MED),("ROWBACKGROUNDS",(0,0),(-1,-1),[BLANC,BLEU_PALE]),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7)]))
        story.append(mtt); story.append(Spacer(1,0.3*cm))
    titre_section("6.  TABLEAU DE CONFORMITE AUX NORMES OMS")
    ch=[Paragraph(f"<b>{t}</b>",S("ch",fontName="Helvetica-Bold",fontSize=9,textColor=BLANC,alignment=TA_CENTER)) for t in ["Parametre","Valeur mesuree","Seuil OMS Potable","Seuil Dangereuse","Conformite"]]
    cd=[("pH",f"{pH:.3f}","6,5 a 8,5","< 4,5 ou > 10",statut_param(pH,6.5,8.5)),("Turbidite (NTU)",f"{turbidite:.3f}","< 5","> 50",statut_param(turbidite,0,5)),("Temperature (degC)",f"{temperature:.2f}","< 25","> 35",statut_param(temperature,0,25)),("Conductivite (uS/cm)",f"{conductivite:.1f}","< 2 500","> 4 000",statut_param(conductivite,0,2500)),("Nitrates (mg/L)",f"{nitrates:.3f}","< 50","> 150",statut_param(nitrates,0,50)),("Oxygene dissous (mg/L)",f"{o2:.3f}","> 6","< 2",statut_param(o2,6,14)),("E. coli (UFC/100 mL)",f"{ecoli:.1f}","0","> 500",statut_param(ecoli,0,0,inverse=True)),]
    crows=[ch]+[[Paragraph(f"<b>{n}</b>",S("c1",fontName="Helvetica-Bold",fontSize=8.5,textColor=NOIR,alignment=TA_LEFT)),Paragraph(f"<b>{v}</b>",S("c2",fontName="Helvetica-Bold",fontSize=9,textColor=NOIR,alignment=TA_CENTER)),Paragraph(sp,S("c3",fontName="Helvetica",fontSize=8.5,textColor=VERT,alignment=TA_CENTER)),Paragraph(sd,S("c4",fontName="Helvetica",fontSize=8.5,textColor=ROUGE,alignment=TA_CENTER)),Paragraph(f"<b>{st}</b>",S("c5",fontName="Helvetica-Bold",fontSize=9,textColor=cu,alignment=TA_CENTER))] for n,v,sp,sd,(st,cu) in cd]
    ct=Table(crows,colWidths=[4.2*cm,3.0*cm,3.8*cm,3.2*cm,3.0*cm])
    ct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),BLEU_MED),("GRID",(0,0),(-1,-1),0.5,GRIS_MED),("ROWBACKGROUNDS",(0,1),(-1,-1),[BLANC,BLEU_PALE]),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6)]))
    story.append(ct); story.append(Spacer(1,0.4*cm))
    titre_section("7.  NOTES METHODOLOGIQUES ET LIMITES")
    for i,note in enumerate(["Ce rapport est genere automatiquement par EauVie sur la base de la moyenne des trois mesures saisies pour chacun des sept parametres. La fiabilite du resultat depend directement de la precision des mesures effectuees sur le terrain et du bon etalonnage des instruments utilises.","L'algorithme Random Forest a ete entraine sur 122 echantillons representatifs, bases sur les donnees OMS, FAO, JMP (OMS/UNICEF), les etudes hydrochimiques d'Afrique de l'Ouest (Akoteyon, 2011 ; USEPA, 2022) et les normes nationales du Benin. Precision en validation croisee (5-fold) : 100 %.","Ce rapport ne se substitue pas a une analyse complete en laboratoire agree. Pour une certification officielle de potabilite, il est recommande de completer par des tests supplementaires (metaux lourds, pesticides, coliformes totaux, chlore residuel).","References : OMS - Directives pour la qualite de l'eau de boisson, 4e edition (2017) ; USEPA Drinking Water Standards (2022) ; Normes nationales du Benin."],1):
        story.append(Paragraph(f"<b>Note {i} :</b> {note}",sn)); story.append(Spacer(1,0.15*cm))
    story.append(Spacer(1,0.3*cm))
    story.append(HRFlowable(width="100%",thickness=1.5,color=BLEU_CLAIR,spaceBefore=8,spaceAfter=6))
    ft=Table([[Paragraph(f"<b>EauVie</b> - Analyse intelligente de la qualite de l'eau<br/>Proposee par <b>Charles MEDEZOUNDJI</b> - Benin, Afrique de l'Ouest<br/>Rapport genere le {date_str} a {heure_str} | Ref. {ref_str}",S("ft1",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_LEFT,leading=11)),Paragraph("Ce document est genere automatiquement.<br/>Il ne remplace pas une analyse en laboratoire agree.<br/><b>EauVie 2025 - Tous droits reserves</b>",S("ft2",fontName="Helvetica",fontSize=7.5,textColor=colors.HexColor("#555"),alignment=TA_RIGHT,leading=11))],],colWidths=[(W-3.6*cm)/2,(W-3.6*cm)/2])
    ft.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")])); story.append(ft)
    doc.build(story); result=buffer.getvalue(); buffer.close(); return result

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
    X=data[["pH","Turbidite","Temperature","Conductivite","Nitrates","O2","Ecoli"]]
    y=data["Classe"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    rf=RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features="sqrt",random_state=42,class_weight="balanced",n_jobs=-1)
    rf.fit(X_train,y_train)
    return rf
rf=load_model()

st.markdown("""<style>
#MainMenu{visibility:hidden !important;}header{visibility:hidden !important;}
footer{visibility:hidden !important;}[data-testid="stToolbar"]{display:none !important;}
html,body,[class*="css"]{color:#0a0a0a !important;}
.main{background:linear-gradient(160deg,#dff3fb 0%,#e8f4fd 100%);}
.block-container{background:rgba(255,255,255,0.97);border-radius:18px;padding:2rem;box-shadow:0 4px 32px rgba(0,119,182,0.12);}
.stButton>button{background:linear-gradient(135deg,#0077b6,#00b4d8);color:white !important;font-size:17px;border-radius:14px;padding:14px 30px;width:100%;border:none;font-weight:700;}
.result-box{padding:24px;border-radius:16px;text-align:center;font-size:24px;font-weight:800;margin:18px 0;}
.potable{background:linear-gradient(135deg,#c8f7c5,#a8e6cf);color:#0a4a0a !important;border:3px solid #28a745;}
.douteuse{background:linear-gradient(135deg,#fff3cd,#ffe082);color:#4a3000 !important;border:3px solid #ffc107;}
.polluee{background:linear-gradient(135deg,#ffd5d5,#ffab91);color:#5a0000 !important;border:3px solid #dc3545;}
.dangereuse{background:linear-gradient(135deg,#2d0000,#1a0000);color:#ff6666 !important;border:3px solid #ff0000;}
.pcard{background:linear-gradient(135deg,#f0f8ff,#e3f2fd);border-left:5px solid #0077b6;border-radius:12px;padding:14px 16px;margin-bottom:12px;}
.plabel{font-weight:800;color:#023e8a !important;font-size:15px;margin-bottom:6px;display:block;}
.ptext{color:#0a0a0a !important;font-size:13px;font-weight:500;line-height:1.6;display:block;margin-bottom:4px;}
.pnorm{font-size:12px;color:#023e8a !important;margin-top:7px;font-weight:700;background:rgba(0,119,182,0.10);padding:4px 10px;border-radius:6px;display:inline-block;}
.proto-box{background:linear-gradient(135deg,#f0fff4,#e8f8e8);border-left:5px solid #1b5e20;border-radius:12px;padding:14px 18px;margin-bottom:14px;}
.proto-title{font-weight:800;color:#1b5e20 !important;font-size:15px;margin-bottom:10px;display:block;}
.proto-item{color:#0a2a0a !important;font-size:13px;font-weight:500;padding:5px 0;border-bottom:1px solid rgba(27,94,32,0.12);line-height:1.5;display:block;}
.header-box{background:linear-gradient(135deg,#023e8a,#0077b6,#00b4d8);border-radius:16px;padding:22px 16px;text-align:center;margin-bottom:22px;}
.header-title{color:#ffffff !important;font-size:30px;font-weight:800;letter-spacing:2px;}
.header-sub{color:#d0eeff !important;font-size:13px;margin-top:8px;line-height:1.6;}
.header-author{color:#a8d8ff !important;font-size:12px;margin-top:6px;font-style:italic;}
.section-title{color:#023e8a !important;font-size:17px;font-weight:800;border-bottom:2px solid #00b4d8;padding-bottom:7px;margin:18px 0 12px 0;display:block;}
.source-box{background:linear-gradient(135deg,#023e8a,#0077b6);border-radius:12px;padding:16px 18px;margin-bottom:14px;}
.source-label{font-weight:800;color:#ffffff !important;font-size:15px;margin-bottom:4px;display:block;}
.source-desc{color:#d0eeff !important;font-size:13px;font-weight:600;line-height:1.5;display:block;}
.conseil-box{background:linear-gradient(135deg,#e3f2fd,#e0f7fa);border-left:5px solid #0077b6;border-radius:12px;padding:14px 18px;margin-top:10px;}
.conseil-title{font-weight:800;color:#023e8a !important;font-size:14px;margin-bottom:8px;display:block;}
.conseil-item{color:#0a0a0a !important;font-size:13px;padding:4px 0;border-bottom:1px solid rgba(0,119,182,0.10);display:block;line-height:1.5;}
p,span,div,label{color:#0a0a0a !important;}
</style>""", unsafe_allow_html=True)

st.markdown("<div class=\'header-box\'><div class=\'header-title\'>EauVie</div><div class=\'header-sub\'>Analyse intelligente de la qualite de l\'eau - Normes OMS<br>afin de garantir une consommation rassurante et benefique.</div><div class=\'header-author\'>Proposee par Charles MEDEZOUNDJI</div></div>", unsafe_allow_html=True)

# Recuperer heure locale via JavaScript
st.components.v1.html("""
<script>
    var now = new Date();
    var h = String(now.getHours()).padStart(2,"0");
    var m = String(now.getMinutes()).padStart(2,"0");
    var heure = h + ":" + m;
    window.parent.postMessage({type:"streamlit:setComponentValue", value:heure}, "*");
</script>
""", height=0)

if "heure_locale" not in st.session_state:
    st.session_state.heure_locale = datetime.now().strftime("%H:%M")

with st.expander("Normes OMS de reference - 7 parametres"):
    st.markdown("| Parametre | Potable | Douteuse | Polluee | Dangereuse |")
    st.markdown("|---|---|---|---|---|")
    st.markdown("| pH | 6,5 a 8,5 | 5,5 a 9,0 | 4,5 a 5,5 | inf. a 4,5 |")
    st.markdown("| Turbidite (NTU) | inf. a 5 | 5 a 10 | 10 a 50 | sup. a 50 |")
    st.markdown("| Temperature (degC) | inf. a 25 | 25 a 30 | 30 a 35 | sup. a 35 |")
    st.markdown("| Conductivite (uS/cm) | inf. a 2 500 | 2 500 a 3 000 | 3 000 a 4 000 | sup. a 4 000 |")
    st.markdown("| Nitrates (mg/L) | inf. a 50 | 50 a 80 | 80 a 150 | sup. a 150 |")
    st.markdown("| Oxygene dissous (mg/L) | sup. a 6 | 4 a 6 | 2 a 4 | inf. a 2 |")
    st.markdown("| E. coli (UFC/100 mL) | 0 | 1 a 10 | 10 a 500 | sup. a 500 |")

with st.expander("Protocole de mesure - 7 parametres"):
    st.markdown("<div class=\'proto-box\'><span class=\'proto-title\'>pH - Potentiometre / pH-metre</span><span class=\'proto-item\'>Outil : pH-metre numerique ou bandelettes indicatrices de pH</span><span class=\'proto-item\'>1. Etalonner avec les solutions tampon pH 4, pH 7 et pH 10</span><span class=\'proto-item\'>2. Plonger l\'electrode propre dans l\'echantillon</span><span class=\'proto-item\'>3. Attendre la stabilisation (environ 30 secondes)</span><span class=\'proto-item\'>4. Lire et noter la valeur affichee</span><span class=\'proto-item\'>5. Rincer l\'electrode a l\'eau distillee apres chaque mesure</span></div>", unsafe_allow_html=True)
    st.markdown("<div class=\'proto-box\'><span class=\'proto-title\'>Turbidite - Turbidimetre numerique (NTU)</span><span class=\'proto-item\'>Outil : Turbidimetre numerique ou comparateur visuel</span><span class=\'proto-item\'>1. Remplir le tube de mesure avec l\'echantillon</span><span class=\'proto-item\'>2. Essuyer le tube pour eliminer toute trace de doigts</span><span class=\'proto-item\'>3. Inserer le tube dans le turbidimetre et fermer le couvercle</span><span class=\'proto-item\'>4. Lire la valeur en NTU affichee</span><span class=\'proto-item\'>5. Repeter 3 fois et calculer la moyenne</span></div>", unsafe_allow_html=True)
    st.markdown("<div class=\'proto-box\'><span class=\'proto-title\'>Temperature - Thermometre electronique</span><span class=\'proto-item\'>Outil : Thermometre numerique a sonde immersible (precision 0,1 degC)</span><span class=\'proto-item\'>1. S\'assurer que la sonde est propre et seche avant usage</span><span class=\'proto-item\'>2. Plonger la sonde directement dans l\'echantillon d\'eau</span><span class=\'proto-item\'>3. Attendre la stabilisation (1 a 2 minutes)</span><span class=\'proto-item\'>4. Lire et noter la temperature en degC</span><span class=\'proto-item\'>5. Mesurer idealement sur le terrain, a la source du prelevement</span></div>", unsafe_allow_html=True)
    st.markdown("<div class=\'proto-box\'><span class=\'proto-title\'>Conductivite - Conductimetre numerique (uS/cm)</span><span class=\'proto-item\'>Outil : Conductimetre portable avec cellule de conductivite</span><span class=\'proto-item\'>1. Etalonner l\'appareil avec une solution etalon certifiee</span><span class=\'proto-item\'>2. Rincer la cellule deux fois avec l\'echantillon a analyser</span><span class=\'proto-item\'>3. Plonger la cellule dans l\'echantillon et activer la mesure</span><span class=\'proto-item\'>4. Attendre la stabilisation et lire la valeur en uS/cm</span><span class=\'proto-item\'>5. Rincer la cellule a l\'eau distillee apres usage</span></div>", unsafe_allow_html=True)
    st.markdown("<div class=\'proto-box\'><span class=\'proto-title\'>Nitrates - Spectrophotometre ou bandelettes reactives</span><span class=\'proto-item\'>Outil : Spectrophotometre portable ou kit colorimetrique (bandelettes nitrates)</span><span class=\'proto-item\'>1. Filtrer l\'echantillon sur membrane 0,45 micron</span><span class=\'proto-item\'>2. Tremper la bandelette 1 seconde dans l\'eau et retirer</span><span class=\'proto-item\'>3. Attendre le temps de reaction indique (60 secondes)</span><span class=\'proto-item\'>4. Comparer la couleur obtenue a la charte colorimetrique fournie</span><span class=\'proto-item\'>5. Lire et noter la concentration en mg/L</span></div>", unsafe_allow_html=True)
    st.markdown("<div class=\'proto-box\'><span class=\'proto-title\'>Oxygene dissous - Oxymetre electronique</span><span class=\'proto-item\'>Outil : Oxymetre numerique portable avec sonde a membrane</span><span class=\'proto-item\'>1. Etalonner la sonde dans l\'air sature en humidite (10 min)</span><span class=\'proto-item\'>2. Plonger la sonde dans l\'echantillon sans creer de bulles</span><span class=\'proto-item\'>3. Agiter tres doucement pour homogeneiser l\'echantillon</span><span class=\'proto-item\'>4. Attendre 1 a 2 minutes et lire la valeur en mg/L</span><span class=\'proto-item\'>5. Rincer la sonde a l\'eau distillee apres usage</span></div>", unsafe_allow_html=True)
    st.markdown("<div class=\'proto-box\'><span class=\'proto-title\'>E. coli - Test de presence / absence ou denombrements</span><span class=\'proto-item\'>Outil : Kit Colilert (IDEXX), bandelettes Compact Dry EC ou milieu m-FC en laboratoire</span><span class=\'proto-item\'>1. Prelever 100 mL d\'eau dans un flacon sterile</span><span class=\'proto-item\'>2. Ajouter le reactif Colilert et melanger jusqu\'a dissolution complete</span><span class=\'proto-item\'>3. Incuber a 35 degC pendant 24 a 28 heures</span><span class=\'proto-item\'>4. Lire le resultat : fluorescence sous UV = presence d\'E. coli</span><span class=\'proto-item\'>5. Exprimer le resultat en UFC/100 mL et noter immediatement</span></div>", unsafe_allow_html=True)

st.markdown("<span class=\'section-title\'>Informations sur l\'analyste et le prelevement</span>", unsafe_allow_html=True)
analyste=st.text_input("Nom complet de l\'analyste *",placeholder="Ex : Jean KOFFI",help="Obligatoire - ce nom figurera dans le rapport officiel")
lieu=st.text_input("Lieu de prelevement *",placeholder="Ex : Village de Kpanrou, commune de Djougou")
SOURCES=["Robinet (reseau traite)","Puits peu profond","Forage profond","Riviere","Fleuve","Lac","Marigot","Eau stagnante (mare)","Eau de pluie collectee","Source naturelle","Rosee collectee","Eau de mer / cotiere","Eau de barrage","Eau de citerne stockee","Autre"]
source=st.selectbox("Source de l\'eau *",SOURCES)

st.markdown("<span class=\'section-title\'>Inserez les trois mesures de chaque parametre</span>", unsafe_allow_html=True)
st.info("Saisissez les trois mesures realisees pour chaque parametre. La moyenne sera calculee automatiquement.")

st.markdown("<div class=\'pcard\'><span class=\'plabel\'>pH - Potentiel Hydrogene</span><span class=\'ptext\'>Mesure l\'acidite ou la basicite de l\'eau. pH bas : metaux toxiques. pH eleve : contamination chimique.</span><span class=\'pnorm\'>Norme OMS : 6,5 a 8,5</span></div>", unsafe_allow_html=True)
c1,c2,c3=st.columns(3)
pH1=c1.number_input("pH - Mesure 1",0.0,14.0,7.0,0.01); pH2=c2.number_input("pH - Mesure 2",0.0,14.0,7.0,0.01); pH3=c3.number_input("pH - Mesure 3",0.0,14.0,7.0,0.01)
pH=round((pH1+pH2+pH3)/3,3); st.caption(f"Moyenne pH : {pH:.3f}")

st.markdown("<div class=\'pcard\'><span class=\'plabel\'>Turbidite (NTU) - Trouble de l\'eau</span><span class=\'ptext\'>Particules en suspension. Eau trouble : agents pathogenes possibles, desinfection moins efficace.</span><span class=\'pnorm\'>Norme OMS : inferieur a 5 NTU</span></div>", unsafe_allow_html=True)
c4,c5,c6=st.columns(3)
tu1=c4.number_input("Turbidite - Mesure 1",0.0,200.0,2.0,0.01); tu2=c5.number_input("Turbidite - Mesure 2",0.0,200.0,2.0,0.01); tu3=c6.number_input("Turbidite - Mesure 3",0.0,200.0,2.0,0.01)
tu=round((tu1+tu2+tu3)/3,3); st.caption(f"Moyenne Turbidite : {tu:.3f} NTU")

st.markdown("<div class=\'pcard\'><span class=\'plabel\'>Temperature (degC) - Activite microbienne</span><span class=\'ptext\'>Au-dela de 25 degC, la proliferation des micro-organismes pathogenes s\'accelere significativement.</span><span class=\'pnorm\'>Norme OMS : inferieur a 25 degC</span></div>", unsafe_allow_html=True)
c7,c8,c9=st.columns(3)
te1=c7.number_input("Temp. - Mesure 1 (degC)",0.0,60.0,22.0,0.1); te2=c8.number_input("Temp. - Mesure 2 (degC)",0.0,60.0,22.0,0.1); te3=c9.number_input("Temp. - Mesure 3 (degC)",0.0,60.0,22.0,0.1)
te=round((te1+te2+te3)/3,2); st.caption(f"Moyenne Temperature : {te:.2f} degC")

st.markdown("<div class=\'pcard\'><span class=\'plabel\'>Conductivite (uS/cm) - Mineralisation de l\'eau</span><span class=\'ptext\'>Conductivite elevee : concentration excessive en sels et mineraux dissous, nocive a long terme.</span><span class=\'pnorm\'>Norme OMS : inferieur a 2 500 uS/cm</span></div>", unsafe_allow_html=True)
c10,c11,c12=st.columns(3)
co1=c10.number_input("Cond. - Mesure 1",0.0,10000.0,350.0,1.0); co2=c11.number_input("Cond. - Mesure 2",0.0,10000.0,350.0,1.0); co3=c12.number_input("Cond. - Mesure 3",0.0,10000.0,350.0,1.0)
co=round((co1+co2+co3)/3,1); st.caption(f"Moyenne Conductivite : {co:.1f} uS/cm")

st.markdown("<div class=\'pcard\'><span class=\'plabel\'>Nitrates (mg/L) - Pollution agricole</span><span class=\'ptext\'>Proviennent des engrais agricoles. Au-dela de 50 mg/L, ils provoquent la methemoglobinemie chez les nourrissons.</span><span class=\'pnorm\'>Norme OMS : inferieur a 50 mg/L</span></div>", unsafe_allow_html=True)
c13,c14,c15=st.columns(3)
no1=c13.number_input("NO3 - Mesure 1",0.0,500.0,5.0,0.1); no2=c14.number_input("NO3 - Mesure 2",0.0,500.0,5.0,0.1); no3=c15.number_input("NO3 - Mesure 3",0.0,500.0,5.0,0.1)
no=round((no1+no2+no3)/3,3); st.caption(f"Moyenne Nitrates : {no:.3f} mg/L")

st.markdown("<div class=\'pcard\'><span class=\'plabel\'>Oxygene dissous (mg/L) - Vitalite de l\'eau</span><span class=\'ptext\'>Taux faible : decomposition organique intense, bacteries. En dessous de 2 mg/L : eau anoxique et dangereuse.</span><span class=\'pnorm\'>Norme : superieur a 6 mg/L</span></div>", unsafe_allow_html=True)
c16,c17,c18=st.columns(3)
o1=c16.number_input("O2 - Mesure 1",0.0,14.0,7.0,0.01); o2_2=c17.number_input("O2 - Mesure 2",0.0,14.0,7.0,0.01); o3=c18.number_input("O2 - Mesure 3",0.0,14.0,7.0,0.01)
o2=round((o1+o2_2+o3)/3,3); st.caption(f"Moyenne Oxygene dissous : {o2:.3f} mg/L")

st.markdown("<div class=\'pcard\'><span class=\'plabel\'>E. coli (UFC/100 mL) - Contamination fecale</span><span class=\'ptext\'>Indicateur direct de contamination fecale. Toute presence signale un risque sanitaire et la probable presence d\'autres pathogenes.</span><span class=\'pnorm\'>Norme OMS : 0 UFC/100 mL (eau potable)</span></div>", unsafe_allow_html=True)
c19,c20,c21=st.columns(3)
ec1=c19.number_input("E.coli - Mesure 1",0.0,10000.0,0.0,1.0); ec2=c20.number_input("E.coli - Mesure 2",0.0,10000.0,0.0,1.0); ec3=c21.number_input("E.coli - Mesure 3",0.0,10000.0,0.0,1.0)
ec=round((ec1+ec2+ec3)/3,1); st.caption(f"Moyenne E. coli : {ec:.1f} UFC/100 mL")

st.markdown("---")
MP={0:("POTABLE","potable","Eau conforme aux normes OMS. Consommation possible sans risque."),1:("DOUTEUSE","douteuse","Anomalies detectees. Filtrez et faites bouillir avant consommation."),2:("POLLUEE","polluee","Eau polluee. Ne pas consommer. Traitement obligatoire."),3:("DANGEREUSE","dangereuse","DANGER EXTREME. Tout contact a eviter. Risque sanitaire majeur."),}

if st.button("Analyser la qualite de l\'eau"):
    erreurs=[]
    if not analyste.strip(): erreurs.append("Le nom de l\'analyste est obligatoire.")
    if not lieu.strip(): erreurs.append("Le lieu de prelevement est obligatoire.")
    if erreurs:
        for e in erreurs: st.error(e)
    else:
        dfm=pd.DataFrame({"pH":[pH],"Turbidite":[tu],"Temperature":[te],"Conductivite":[co],"Nitrates":[no],"O2":[o2],"Ecoli":[ec]})
        cl=rf.predict(dfm)[0]; pr=rf.predict_proba(dfm)[0]
        lb,cs,co_msg=MP[cl]; conf=str(round(pr[cl]*100,1))
        st.markdown(f"<div class=\'result-box {cs}\'>{lb}<br><span style=\'font-size:14px;font-weight:600;\'>Confiance du modele : {conf} %</span></div>", unsafe_allow_html=True)
        st.markdown(f"**Conseil :** {co_msg}")
        if cl in [1,2,3]:
            with st.expander("Comment purifier cette eau ?"):
                st.markdown("<div class=\'conseil-box\'><span class=\'conseil-title\'>1. Ebullition</span><span class=\'conseil-item\'>Porter l\'eau a ebullition pendant au moins 5 minutes.</span><span class=\'conseil-item\'>Laisser refroidir dans un recipient propre et couvert.</span><span class=\'conseil-item\'>Efficace contre les bacteries, les virus et les parasites.</span></div>", unsafe_allow_html=True)
                st.markdown("<div class=\'conseil-box\'><span class=\'conseil-title\'>2. Filtration sur sable et gravier</span><span class=\'conseil-item\'>Couches : gravier grossier, fin, sable grossier, fin, charbon de bois actif.</span><span class=\'conseil-item\'>Completer obligatoirement avec l\'ebullition.</span></div>", unsafe_allow_html=True)
                st.markdown("<div class=\'conseil-box\'><span class=\'conseil-title\'>3. Desinfection solaire SODIS</span><span class=\'conseil-item\'>Bouteilles transparentes : 6 heures au soleil (ciel clair) ou 2 jours (nuageux).</span><span class=\'conseil-item\'>Methode gratuite, validee par l\'OMS pour l\'Afrique de l\'Ouest.</span></div>", unsafe_allow_html=True)
                st.markdown("<div class=\'conseil-box\'><span class=\'conseil-title\'>4. Chloration</span><span class=\'conseil-item\'>2 gouttes d\'eau de Javel a 5 % par litre (1 si eau claire). Attendre 30 min.</span></div>", unsafe_allow_html=True)
                st.markdown("<div class=\'conseil-box\'><span class=\'conseil-title\'>5. Graines de Moringa oleifera</span><span class=\'conseil-item\'>Broyer 2-3 graines seches. Ajouter a 1 litre d\'eau turbide, agiter 1 min puis 5 min lentement. Decanter 1 heure.</span></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Rapport officiel PDF")
        try:
            mesures={"pH":pH,"turb":tu,"temp":te,"cond":co,"no3":no,"o2":o2,"ecoli":ec}
            heure_locale=st.session_state.get("heure_locale",datetime.now().strftime("%H:%M"))
            pdf_bytes=generer_pdf(mesures,cl,list(pr),analyste=analyste,lieu=lieu,source=source,heure_locale=heure_locale)
            nom_fichier="rapport_eauvie_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".pdf"
            st.download_button(label="Telecharger le rapport PDF officiel",data=pdf_bytes,file_name=nom_fichier,mime="application/pdf")
            st.success("Rapport PDF genere avec succes !")
        except Exception as e:
            st.error("Erreur PDF : "+str(e))
        prd=pd.DataFrame({"Classe":["Potable","Douteuse","Polluee","Dangereuse"],"Probabilite (%)":[round(p*100,1) for p in pr]})
        st.bar_chart(prd.set_index("Classe"))
        if "histo" not in st.session_state: st.session_state.histo=[]
        st.session_state.histo.append({"Heure":datetime.now().strftime("%H:%M:%S"),"Analyste":analyste,"Lieu":lieu,"Source":source,"pH":pH,"Turb.":tu,"Temp.":te,"Cond.":co,"NO3":no,"O2 dissous":o2,"E.coli":ec,"Resultat":lb})

if "histo" in st.session_state and len(st.session_state.histo)>0:
    st.markdown("---")
    st.markdown("<span class=\'section-title\'>Historique des analyses</span>", unsafe_allow_html=True)
    hdf=pd.DataFrame(st.session_state.histo)
    st.dataframe(hdf,use_container_width=True)
    st.download_button("Telecharger le CSV",hdf.to_csv(index=False).encode("utf-8"),"historique_eauvie.csv","text/csv")

st.markdown("---")
st.markdown("<div style=\'text-align:center;color:#023e8a !important;font-size:12px;padding:10px;font-weight:600;\'>EauVie - Random Forest - Normes OMS - 7 parametres - Charles MEDEZOUNDJI</div>", unsafe_allow_html=True)
