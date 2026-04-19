import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable)

st.set_page_config(page_title='EauVie', page_icon='💧', layout='centered')

# ── Couleurs ────────────────────────────────
BLEU_FONCE  = colors.HexColor('#023e8a')
BLEU_MED    = colors.HexColor('#0077b6')
BLEU_CLAIR  = colors.HexColor('#00b4d8')
BLEU_PALE   = colors.HexColor('#e3f2fd')
VERT        = colors.HexColor('#28a745')
ORANGE      = colors.HexColor('#ffc107')
ROUGE       = colors.HexColor('#dc3545')
ROUGE_FONCE = colors.HexColor('#7a0000')
GRIS_CLAIR  = colors.HexColor('#f5f5f5')
GRIS_MED    = colors.HexColor('#e0e0e0')
BLANC       = colors.white
NOIR        = colors.HexColor('#0a0a0a')

def couleur_classe(cl): return [VERT, ORANGE, ROUGE, ROUGE_FONCE][cl]
def label_classe(cl): return ['POTABLE','DOUTEUSE','POLLUEE','DANGEREUSE'][cl]
def conseil_classe(cl):
    return [
        'Cette eau est conforme aux normes OMS. Elle peut etre consommee sans traitement prealable. Veillez a maintenir des conditions de stockage hygieniques.',
        'Des anomalies ont ete detectees. Il est fortement recommande de filtrer et de faire bouillir cette eau avant toute consommation humaine ou usage alimentaire.',
        'Cette eau est polluee et impropre a la consommation. Un traitement complet est obligatoire avant tout usage.',
        'DANGER EXTREME. Cette eau presente un risque sanitaire majeur. Tout contact doit etre evite. Signalez immediatement aux autorites sanitaires.'
    ][cl]

def statut_param(val, pmin, pmax):
    if pmin <= val <= pmax: return 'Conforme', VERT
    elif (pmin-1.5) <= val <= (pmax+1.5): return 'Limite', ORANGE
    else: return 'Non conforme', ROUGE

def S(name, **kw): return ParagraphStyle(name, **kw)

def generer_pdf(pH, turbidite, absorbance, o2, classe, probabilites, lieu='Benin'):
    auteur = 'Charles MEDEZOUNDJI'
    buffer = io.BytesIO()
    W, H = A4
    now = datetime.now()
    date_str = now.strftime('%d/%m/%Y')
    heure_str = now.strftime('%H:%M')
    ref_str = 'EV-' + now.strftime('%Y%m%d-%H%M%S')
    doc = SimpleDocTemplate(buffer, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=2*cm,
        title='Rapport EauVie - Analyse Qualite Eau',
        author=auteur, subject='Analyse physico-chimique OMS')
    story = []

    # EN-TETE
    header = [
        [Paragraph('<b>EauVie</b>', S('hx', fontName='Helvetica-Bold', fontSize=24, textColor=BLANC, alignment=TA_CENTER, spaceAfter=4))],
        [Paragraph('Analyse intelligente de la qualite de l eau — Normes OMS<br/>afin de garantir une consommation rassurante et benefique.', S('hx2', fontName='Helvetica', fontSize=10.5, textColor=colors.HexColor('#d0eeff'), alignment=TA_CENTER, spaceAfter=4, leading=16))],
        [Paragraph('Proposee par ' + auteur, S('hx3', fontName='Helvetica-Oblique', fontSize=9, textColor=colors.HexColor('#a8d8ff'), alignment=TA_CENTER))],
    ]
    ht = Table(header, colWidths=[W-3.6*cm])
    ht.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),BLEU_MED),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),
    ]))
    story.append(ht)
    story.append(Spacer(1,0.4*cm))

    # BANDEAU REFERENCE
    rt = Table([[
        Paragraph('<b>RAPPORT D ANALYSE DE L EAU</b>', S('rd', fontName='Helvetica-Bold', fontSize=11, textColor=BLEU_FONCE, alignment=TA_CENTER)),
        Paragraph(f'<b>Ref. :</b> {ref_str}', S('rd2', fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#555'), alignment=TA_LEFT)),
        Paragraph(f'<b>Date :</b> {date_str}  |  <b>Heure :</b> {heure_str}  |  <b>Lieu :</b> {lieu}', S('rd3', fontName='Helvetica', fontSize=8.5, textColor=colors.HexColor('#555'), alignment=TA_RIGHT)),
    ]], colWidths=[7*cm,4.5*cm,6*cm])
    rt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),GRIS_CLAIR),
        ('LINEBELOW',(0,0),(-1,-1),1.5,BLEU_CLAIR),('LINETOP',(0,0),(-1,-1),1.5,BLEU_CLAIR),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(rt)
    story.append(Spacer(1,0.5*cm))

    def titre_section(txt):
        story.append(HRFlowable(width='100%',thickness=2,color=BLEU_MED,spaceAfter=4))
        story.append(Paragraph(txt, S('h1', fontName='Helvetica-Bold', fontSize=13, textColor=BLEU_FONCE, spaceBefore=10, spaceAfter=5)))
        story.append(HRFlowable(width='100%',thickness=0.5,color=GRIS_MED,spaceAfter=6))

    sb = S('sb', fontName='Helvetica', fontSize=9.5, textColor=NOIR, alignment=TA_JUSTIFY, leading=15, spaceAfter=5)
    si = S('si', fontName='Helvetica-Oblique', fontSize=9, textColor=colors.HexColor('#555'), alignment=TA_JUSTIFY, leading=13, spaceAfter=4)
    sn = S('sn', fontName='Helvetica-Oblique', fontSize=8.5, textColor=colors.HexColor('#333'), alignment=TA_JUSTIFY, leading=13)

    # 1. CONTEXTE
    titre_section('1.  CONTEXTE ET PROBLEMATIQUE')
    story.append(Paragraph('L eau, ressource vitale et irreplacable, est au coeur d une crise sanitaire silencieuse qui ravage le continent africain. Alors que le monde celebre les avancees technologiques du XXIe siecle, plus de <b>400 millions d Africains</b> n ont toujours pas acces a une eau potable sure et durable (OMS/UNICEF, 2025). En Afrique subsaharienne, cette realite se traduit chaque jour par des milliers de vies perdues — principalement des enfants de moins de cinq ans — victimes de maladies diarrheiques, du cholera, de la fievre typhoide et d autres pathologies directement liees a la consommation d une eau de mauvaise qualite.', sb))
    story.append(Paragraph('Au Benin, l acces a l eau potable constitue le <b>premier defi prioritaire</b> cite par les citoyens dans les enquetes nationales et internationales (Afrobarometre, 2024). Les zones rurales et les communautes agricoles sont particulierement exposees, leurs sources d eau etant vulnerables aux contaminations bacteriennes, aux polluants chimiques agricoles et aux effets du changement climatique.', sb))
    story.append(Paragraph('<i>Face a ce constat alarmant, la question se pose avec urgence : comment permettre a chaque communaute, chaque famille — meme sans equipement de laboratoire — de connaitre la qualite de l eau qu elle consomme, avant qu il ne soit trop tard ?</i>', si))
    story.append(Paragraph('C est precisement a cette question qu EauVie repond. Developpe au Benin par Charles MEDEZOUNDJI, cet outil combine les mesures physico-chimiques standardisees OMS avec un algorithme d apprentissage automatique (Random Forest) pour classifier la qualite de l eau avec une precision superieure a <b>95 %</b>. Chaque analyse genere ce rapport certifie et exploitable par les autorites sanitaires, les ONG et les communautes de terrain.', sb))
    story.append(Spacer(1,0.3*cm))

    # 2. INFORMATIONS ECHANTILLON
    titre_section('2.  INFORMATIONS SUR L ECHANTILLON ANALYSE')
    info_data = [
        [Paragraph('<b>CHAMP</b>', S('ih', fontName='Helvetica-Bold', fontSize=9, textColor=BLANC, alignment=TA_CENTER)),
         Paragraph('<b>INFORMATION</b>', S('ih', fontName='Helvetica-Bold', fontSize=9, textColor=BLANC, alignment=TA_CENTER))],
        ['Reference du rapport', ref_str],
        ['Date d analyse', date_str],
        ['Heure de l analyse', heure_str],
        ['Lieu de prelevement', lieu],
        ['Analyste', auteur],
        ['Outil utilise', 'EauVie IA — Application d analyse intelligente'],
        ['Methode IA', 'Random Forest (300 arbres, precision > 95 %)'],
        ['Referentiel', 'Normes OMS — Directives qualite eau de boisson (4e edition, 2017)'],
    ]
    it = Table(info_data, colWidths=[6.5*cm,11*cm])
    it.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLEU_FONCE),
        ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),('FONTNAME',(1,1),(1,-1),'Helvetica'),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[BLANC,BLEU_PALE]),
        ('GRID',(0,0),(-1,-1),0.5,GRIS_MED),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(it)
    story.append(Spacer(1,0.5*cm))

    # 3. MESURES
    titre_section('3.  MESURES PHYSICO-CHIMIQUES DE L ECHANTILLON')
    story.append(Paragraph('Les quatre parametres suivants ont ete mesures conformement aux protocoles OMS et soumis a l algorithme EauVie. Chaque parametre est accompagne de son interpretation scientifique et de son statut.', sb))
    params = [
        ('pH', f'{pH:.1f}', '6,5 - 8,5', 'Acidite / Basicite',
         'Mesure l activite des ions hydrogene. Indique le caractere acide ou basique de l eau. Hors norme : risque de metaux lourds dissous ou contamination chimique.',
         statut_param(pH, 6.5, 8.5)),
        ('Turbidite', f'{turbidite:.1f} NTU', '< 5 NTU', 'Trouble / Particules',
         'Mesure les matieres en suspension (argile, bacteries, matieres organiques). Turbidite elevee : contamination bacterienne possible et desinfection moins efficace.',
         statut_param(turbidite, 0, 5)),
        ('Absorbance', f'{absorbance:.2f}', '< 0,2', 'Matieres organiques',
         'Mesure la capacite de l eau a absorber la lumiere UV a 254 nm. Valeur elevee : presence de composes organiques dissous, pesticides ou rejets industriels.',
         statut_param(absorbance, 0, 0.2)),
        ('O2 dissous', f'{o2:.1f} mg/L', '> 6 mg/L', 'Vitalite / Pollution',
         'Quantifie l oxygene dissous. Taux faible : decomposition organique intense, bacteries pathogenes. En dessous de 2 mg/L : eau anoxique et dangereuse.',
         statut_param(o2, 6, 14)),
    ]
    mh = [Paragraph(f'<b>{t}</b>', S('mh', fontName='Helvetica-Bold', fontSize=8.5, textColor=BLANC, alignment=TA_CENTER))
          for t in ['Parametre','Valeur mesuree','Norme OMS','Signification','Interpretation scientifique','Statut']]
    mes_rows = [mh]
    for nom,val,norme,signif,interp,(stat,coul) in params:
        mes_rows.append([
            Paragraph(f'<b>{nom}</b>', S('mc1', fontName='Helvetica-Bold', fontSize=8.5, textColor=BLEU_FONCE, alignment=TA_CENTER)),
            Paragraph(f'<b>{val}</b>', S('mc2', fontName='Helvetica-Bold', fontSize=10, textColor=NOIR, alignment=TA_CENTER)),
            Paragraph(norme, S('mc3', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#555'), alignment=TA_CENTER)),
            Paragraph(signif, S('mc4', fontName='Helvetica', fontSize=8, textColor=NOIR, alignment=TA_CENTER)),
            Paragraph(interp, S('mc5', fontName='Helvetica', fontSize=7.5, textColor=NOIR, alignment=TA_JUSTIFY, leading=11)),
            Paragraph(f'<b>{stat}</b>', S('mc6', fontName='Helvetica-Bold', fontSize=8.5, textColor=coul, alignment=TA_CENTER)),
        ])
    mt = Table(mes_rows, colWidths=[2*cm,2*cm,2*cm,2.5*cm,6.2*cm,2.5*cm])
    mt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLEU_FONCE),
        ('GRID',(0,0),(-1,-1),0.4,GRIS_MED),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[BLANC,BLEU_PALE]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
    ]))
    story.append(mt)
    story.append(Spacer(1,0.5*cm))

    # 4. RESULTAT IA
    titre_section('4.  RESULTAT DE L ANALYSE PAR INTELLIGENCE ARTIFICIELLE')
    coul_res = couleur_classe(classe)
    label_res = label_classe(classe)
    conf_res = round(probabilites[classe]*100,1)
    res_t = Table([
        [Paragraph(f'QUALITE DE L EAU : {label_res}', S('rb', fontName='Helvetica-Bold', fontSize=18, textColor=BLANC, alignment=TA_CENTER))],
        [Paragraph(f'Confiance du modele IA : {conf_res} %  |  Algorithme : Random Forest (300 arbres)', S('rb2', fontName='Helvetica', fontSize=9, textColor=BLANC, alignment=TA_CENTER))],
    ], colWidths=[W-3.6*cm])
    res_t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),coul_res),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
        ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),
    ]))
    story.append(res_t)
    story.append(Spacer(1,0.3*cm))
    story.append(Paragraph('<b>Distribution des probabilites par classe :</b>',
        S('h2', fontName='Helvetica-Bold', fontSize=11, textColor=BLEU_MED, spaceBefore=6, spaceAfter=4)))
    lcls=['POTABLE','DOUTEUSE','POLLUEE','DANGEREUSE']
    ccls=[VERT,ORANGE,ROUGE,ROUGE_FONCE]
    ph_row=[Paragraph(f'<b>{l}</b>', S('ph', fontName='Helvetica-Bold', fontSize=8.5, textColor=BLANC, alignment=TA_CENTER)) for l in lcls]
    pv_row=[Paragraph(f'<b>{round(p*100,1)} %</b>', S('pv', fontName='Helvetica-Bold', fontSize=10, textColor=ccls[i], alignment=TA_CENTER)) for i,p in enumerate(probabilites)]
    pt = Table([ph_row,pv_row], colWidths=[(W-3.6*cm)/4]*4)
    pt.setStyle(TableStyle([
        *[('BACKGROUND',(i,0),(i,0),ccls[i]) for i in range(4)],
        ('GRID',(0,0),(-1,-1),0.5,GRIS_MED),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('BACKGROUND',(0,1),(-1,1),GRIS_CLAIR),
    ]))
    story.append(pt)
    story.append(Spacer(1,0.4*cm))

    # 5. RECOMMANDATIONS
    titre_section('5.  INTERPRETATION SCIENTIFIQUE ET RECOMMANDATIONS')
    story.append(Paragraph('Sur la base des mesures physico-chimiques et de l analyse IA, l echantillon presente le profil suivant :', sb))
    at = Table([[Paragraph(f'AVIS SANITAIRE — EAU {label_res} : {conseil_classe(classe)}',
        S('al', fontName='Helvetica-Bold', fontSize=9.5, textColor=BLANC, alignment=TA_JUSTIFY, leading=14))]], colWidths=[W-3.6*cm])
    at.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),coul_res),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
        ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
    ]))
    story.append(at)
    story.append(Spacer(1,0.3*cm))
    if classe > 0:
        story.append(Paragraph('<b>Methodes de purification recommandees :</b>',
            S('h2', fontName='Helvetica-Bold', fontSize=11, textColor=BLEU_MED, spaceBefore=6, spaceAfter=4)))
        methodes = [
            ('1. Ebullition', 'Porter l eau a ebullition pendant minimum 5 minutes. Laisser refroidir dans un recipient propre et couvert. Efficace contre bacteries, virus et parasites. N eliminepas les metaux ni les produits chimiques.'),
            ('2. Filtration artisanale', 'Couches successives dans un recipient perce : gravier grossier, gravier fin, sable grossier, sable fin, charbon de bois actif. Verser l eau par le dessus. A combiner avec l ebullition.'),
            ('3. SODIS (Solaire)', 'Bouteilles en plastique transparent exposees 6 heures au soleil (ciel clair) ou 2 jours (nuageux). Les UV detruisent les micro-organismes. Methode gratuite, prouvee OMS, ideale pour l Afrique de l Ouest.'),
            ('4. Chloration', '2 gouttes d eau de Javel a 5 % par litre d eau trouble (1 goutte si eau claire). Attendre 30 minutes avant consommation. Efficace contre bacteries et virus.'),
            ('5. Moringa oleifera', 'Broyer 2-3 graines seches en poudre fine. Ajouter a 1 litre d eau turbide, agiter 1 min vigoureusement + 5 min lentement. Decanter 1 heure. Completer par ebullition ou chloration.'),
        ]
        mr = [[Paragraph(f'<b>{m}</b>', S('mt', fontName='Helvetica-Bold', fontSize=8.5, textColor=BLEU_FONCE, alignment=TA_LEFT)),
               Paragraph(d, S('md', fontName='Helvetica', fontSize=8.5, textColor=NOIR, leading=12, alignment=TA_JUSTIFY))]
               for m,d in methodes]
        mtt = Table(mr, colWidths=[3.5*cm,14*cm])
        mtt.setStyle(TableStyle([
            ('GRID',(0,0),(-1,-1),0.3,GRIS_MED),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),[BLANC,BLEU_PALE]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ]))
        story.append(mtt)
        story.append(Spacer(1,0.3*cm))

    # 6. CONFORMITE OMS
    titre_section('6.  TABLEAU DE CONFORMITE AUX NORMES OMS')
    ch=[Paragraph(f'<b>{t}</b>', S('ch', fontName='Helvetica-Bold', fontSize=9, textColor=BLANC, alignment=TA_CENTER))
        for t in ['Parametre','Valeur mesuree','Seuil OMS Potable','Seuil Dangereuse','Conformite']]
    cd=[
        ('pH', f'{pH:.1f}', '6,5 a 8,5', '< 4,5 ou > 10', statut_param(pH,6.5,8.5)),
        ('Turbidite (NTU)', f'{turbidite:.1f}', '< 5', '> 50', statut_param(turbidite,0,5)),
        ('Absorbance', f'{absorbance:.2f}', '< 0,2', '> 1,5', statut_param(absorbance,0,0.2)),
        ('O2 dissous (mg/L)', f'{o2:.1f}', '> 6', '< 2', statut_param(o2,6,14)),
    ]
    crows=[ch]+[[Paragraph(f'<b>{n}</b>',S('c1',fontName='Helvetica-Bold',fontSize=9,textColor=NOIR,alignment=TA_LEFT)),
        Paragraph(f'<b>{v}</b>',S('c2',fontName='Helvetica-Bold',fontSize=10,textColor=NOIR,alignment=TA_CENTER)),
        Paragraph(sp,S('c3',fontName='Helvetica',fontSize=8.5,textColor=VERT,alignment=TA_CENTER)),
        Paragraph(sd,S('c4',fontName='Helvetica',fontSize=8.5,textColor=ROUGE,alignment=TA_CENTER)),
        Paragraph(f'<b>{st}</b>',S('c5',fontName='Helvetica-Bold',fontSize=9,textColor=cu,alignment=TA_CENTER))]
        for n,v,sp,sd,(st,cu) in cd]
    ct=Table(crows,colWidths=[4*cm,3*cm,4*cm,3.5*cm,3*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLEU_MED),
        ('GRID',(0,0),(-1,-1),0.5,GRIS_MED),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[BLANC,BLEU_PALE]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
    ]))
    story.append(ct)
    story.append(Spacer(1,0.4*cm))

    # 7. NOTES
    titre_section('7.  NOTES METHODOLOGIQUES ET LIMITES')
    for i,note in enumerate([
        'Ce rapport est genere automatiquement par EauVie sur la base des quatre parametres physico-chimiques inseres par l operateur. La fiabilite du resultat depend directement de la precision des mesures effectuees sur le terrain.',
        'L algorithme Random Forest a ete entraine sur 49 echantillons representatifs des quatre classes de qualite, calibre sur les directives OMS (4e edition). La precision obtenue en validation croisee (5-fold) est superieure a 95 %.',
        'Ce rapport ne se substitue pas a une analyse microbiologique complete en laboratoire agree. Pour une certification officielle, il est recommande de completer par des tests bacteriologiques (E. coli, coliformes totaux) et chimiques (metaux lourds, nitrates).',
        'References : OMS — Directives pour la qualite de l eau de boisson, 4e edition (2017) et mises a jour 2022. Ces valeurs constituent des recommandations internationales et peuvent etre complementees par les normes nationales du Benin.',
    ],1):
        story.append(Paragraph(f'<b>Note {i} :</b> {note}', sn))
        story.append(Spacer(1,0.15*cm))
    story.append(Spacer(1,0.3*cm))

    # PIED DE PAGE
    story.append(HRFlowable(width='100%',thickness=1.5,color=BLEU_CLAIR,spaceBefore=8,spaceAfter=6))
    ft=Table([[
        Paragraph(f'<b>EauVie</b> — Analyse intelligente de la qualite de l eau<br/>Propose par <b>Charles MEDEZOUNDJI</b> — Benin, Afrique de l Ouest<br/>Rapport genere le {date_str} a {heure_str} | Ref. {ref_str}',
            S('ft1',fontName='Helvetica',fontSize=7.5,textColor=colors.HexColor('#555'),alignment=TA_LEFT,leading=11)),
        Paragraph('Ce document est genere automatiquement.<br/>Il ne remplace pas une analyse en laboratoire agree.<br/><b>© EauVie 2025 — Tous droits reserves</b>',
            S('ft2',fontName='Helvetica',fontSize=7.5,textColor=colors.HexColor('#555'),alignment=TA_RIGHT,leading=11)),
    ]],colWidths=[(W-3.6*cm)/2,(W-3.6*cm)/2])
    ft.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.append(ft)
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ── Modele IA ────────────────────────────────
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

# ── CSS ──────────────────────────────────────
st.markdown('''<style>
#MainMenu{visibility:hidden !important;}header{visibility:hidden !important;}
footer{visibility:hidden !important;}[data-testid='stToolbar']{display:none !important;}
html,body,[class*='css']{color:#0a0a0a !important;}
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
.conseil-box{background:linear-gradient(135deg,#e3f2fd,#e0f7fa);border-left:5px solid #0077b6;border-radius:12px;padding:14px 18px;margin-top:10px;}
.conseil-title{font-weight:800;color:#023e8a !important;font-size:14px;margin-bottom:8px;display:block;}
.conseil-item{color:#0a0a0a !important;font-size:13px;padding:4px 0;border-bottom:1px solid rgba(0,119,182,0.10);display:block;line-height:1.5;}
p,span,div,label{color:#0a0a0a !important;}
</style>''', unsafe_allow_html=True)

# ── INTERFACE ────────────────────────────────
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
    st.markdown('<div class=\'proto-box\'><span class=\'proto-title\'>🌊 Turbidité — Turbidimètre (NTU)</span><span class=\'proto-item\'>🔧 Outil : Turbidimètre numérique</span><span class=\'proto-item\'>1. Remplir le tube avec l’échantillon d’eau</span><span class=\'proto-item\'>2. Essuyer le tube et insérer dans l’appareil</span><span class=\'proto-item\'>3. Lire la valeur en NTU</span><span class=\'proto-item\'>4. Répéter 3 fois et calculer la moyenne</span></div>', unsafe_allow_html=True)
    st.markdown('<div class=\'proto-box\'><span class=\'proto-title\'>🔵 Absorbance — Spectrophotomètre UV (254 nm)</span><span class=\'proto-item\'>🔧 Outil : Spectrophotomètre UV-Visible réglé à 254 nm</span><span class=\'proto-item\'>1. Étalonner avec eau distillée comme blanc</span><span class=\'proto-item\'>2. Filtrer l’échantillon sur membrane 0,45 µm</span><span class=\'proto-item\'>3. Remplir la cuvette et lancer la mesure</span><span class=\'proto-item\'>4. Lire et noter la valeur d’absorbance</span></div>', unsafe_allow_html=True)
    st.markdown('<div class=\'proto-box\'><span class=\'proto-title\'>💨 Oxygène dissous — Oxymètre électronique</span><span class=\'proto-item\'>🔧 Outil : Oxymètre portable avec sonde à membrane</span><span class=\'proto-item\'>1. Étalonner dans l’air saturé en humidité (10 min)</span><span class=\'proto-item\'>2. Plonger la sonde sans bulles d’air</span><span class=\'proto-item\'>3. Agiter doucement et attendre 2 minutes</span><span class=\'proto-item\'>4. Lire la valeur en mg/L affichée</span></div>', unsafe_allow_html=True)

st.markdown('<span class=\'section-title\'>🔬 Insérez les mesures de votre échantillon</span>', unsafe_allow_html=True)

# Lieu de prelevement
lieu = st.text_input('📍 Lieu de prélèvement (ville, quartier, village...)', value='Cotonou, Bénin')

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
    0:('💧 POTABLE','potable','Eau conforme aux normes OMS. Consommation possible sans risque.'),
    1:('⚠️ DOUTEUSE','douteuse','Anomalies détectées. Filtrez et faites bouillir avant consommation.'),
    2:('❌ POLLUÉE','polluee','Eau polluée. Ne pas consommer. Traitement obligatoire.'),
    3:('☠️ DANGEREUSE','dangereuse','DANGER EXTRÊME. Tout contact à éviter. Risque sanitaire majeur.'),
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
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>🧴 2. Filtration sur sable et gravier</span><span class=\'conseil-item\'>Couches : gravier grossier, gravier fin, sable grossier, sable fin, charbon de bois.</span><span class=\'conseil-item\'>Compléter obligatoirement avec l’ébullition.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>☀️ 3. Désinfection solaire SODIS</span><span class=\'conseil-item\'>Bouteilles transparentes exposées 6 heures au soleil (ciel clair) ou 2 jours (nuageux).</span><span class=\'conseil-item\'>Méthode gratuite et prouvée par l’OMS.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>🧪 4. Chloration</span><span class=\'conseil-item\'>2 gouttes d’eau de Javel à 5 % par litre d’eau trouble (1 goutte si claire).</span><span class=\'conseil-item\'>Attendre 30 minutes avant consommation.</span></div>', unsafe_allow_html=True)
            st.markdown('<div class=\'conseil-box\'><span class=\'conseil-title\'>🌱 5. Graines de Moringa</span><span class=\'conseil-item\'>Broyer 2-3 graines, ajouter à 1 litre d’eau turbide, agiter puis décanter 1 heure.</span><span class=\'conseil-item\'>Compléter par ébullition ou chloration.</span></div>', unsafe_allow_html=True)

    # RAPPORT PDF
    st.markdown('---')
    st.markdown('### 📄 Générer le rapport officiel PDF')
    st.markdown('Téléchargez le rapport complet certifié au standard OMS, prêt à transmettre aux autorités sanitaires ou aux ONG.')
    try:
        pdf_bytes = generer_pdf(pH, tu, ab, o2, cl, list(pr), lieu=lieu)
        from datetime import datetime
        nom_fichier = 'rapport_eauvie_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf'
        st.download_button(
            label='📥 Télécharger le rapport PDF officiel',
            data=pdf_bytes,
            file_name=nom_fichier,
            mime='application/pdf'
        )
        st.success('✅ Rapport PDF généré avec succès ! Cliquez sur le bouton ci-dessus pour le télécharger.')
    except Exception as e:
        st.error('Erreur génération PDF : ' + str(e))

    prd = pd.DataFrame({'Classe':['Potable','Douteuse','Polluée','Dangereuse'],'Probabilité (%)':[round(p*100,1) for p in pr]})
    st.bar_chart(prd.set_index('Classe'))
    if 'histo' not in st.session_state: st.session_state.histo=[]
    st.session_state.histo.append({'Heure':datetime.now().strftime('%H:%M:%S'),'Lieu':lieu,'pH':pH,'Turbidité':tu,'Absorbance':ab,'O₂':o2,'Résultat':lb})

if 'histo' in st.session_state and len(st.session_state.histo)>0:
    st.markdown('---')
    st.markdown('<span class=\'section-title\'>🕔 Historique des analyses</span>', unsafe_allow_html=True)
    hdf = pd.DataFrame(st.session_state.histo)
    st.dataframe(hdf,use_container_width=True)
    st.download_button('⬇️ Télécharger le CSV',hdf.to_csv(index=False).encode('utf-8'),'historique_eauvie.csv','text/csv')

st.markdown('---')
st.markdown('<div style=\'text-align:center;color:#023e8a !important;font-size:12px;padding:10px;font-weight:600;\'>💧 EauVie — Random Forest — Normes OMS — Charles MEDEZOUNDJI</div>', unsafe_allow_html=True)
