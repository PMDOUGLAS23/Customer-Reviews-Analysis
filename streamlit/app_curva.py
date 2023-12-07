import streamlit as st
import pandas as pd
import io

df = pd.read_csv("redoute.csv", sep=";")

# Titre principal
st.title(":blue[Customers Reviews Analytics]")

# Barre latérale avec le sommaire
st.sidebar.title(":blue[Sommaire]")
pages = [
    "Introduction",
    "Collecte des Données",
    "Exploration et Analyse des Données",
    "Prédiction du Rating",
    "Analyse de Sentiment",
    "Topic Modeling",
    "Prédiction de la Réponse Fournisseur",
    "Démo",
    "Conclusion et Perspectives",
]

# Sélection de la page dans la barre latérale
page = st.sidebar.radio("Aller vers", pages)

# Contenu de chaque page
if page == pages[0]:
    st.write("## Introduction")
    st.write("#### Objectifs du Projet")
    st.markdown("- Prédire les notations des clients")
    st.markdown("- Catégoriser les commentaires")
    st.markdown("- Proposer des réponses automatiques")
    st.markdown("- Analyser les sentiments")

    st.write("#### Equipe Projet")
    st.text(
        """
        xxx
        xxx
        xxx
        """
    )

if page == pages[1]:
    # Source de données
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Source de données")
    with col2:
        st.markdown(
            """ **TrustedShops**, Entreprise allemande qui propose entre autres:
                        Certification de sites web marchands
                        Services **d'évaluation et d'avis clients**
                    """,
            unsafe_allow_html=True,
        )

    # Entreprise cible
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Entreprise ciblée")
    with col4:
        st.markdown(
            "**La Redoute**, Leader français du e-commerce en mode et maison",
            unsafe_allow_html=True,
        )
    col5, col6 = st.columns(2)
    with col5:
        st.markdown(
            "##### Collecte des données des avis clients", unsafe_allow_html=True
        )
    with col6:
        st.markdown(
            "Etape 1 : **request.get + BeautifulSoup + Pandas**", unsafe_allow_html=True
        )
        st.markdown(
            "Étape 2: 1er nettoyage et formatage des données, puis stockage dans un fichier .csv",
            unsafe_allow_html=True,
        )

    st.markdown("##### Données pour l'Analyse Exploratoire", unsafe_allow_html=True)
    st.dataframe(df.head(5))
    # pour afficher df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)


if page == pages[2]:
    st.write("## Exploration et Analyse des Données")
    st.markdown("- xx")
    st.markdown("- xx")
    st.markdown("- xx")

if page == pages[3]:
    st.write("## Prédiction du Rating en Fonction des Commentaires")
    st.write("#### Problématique")
    # poser la problématique

    st.write("#### Démarche")
    # Modélisation de base avec 3 variables explicatives
    # Bag of Words
    # GRU
    st.write("#### Modélisation de Base")
    # Modélisation de base avec 3 variables explicatives

    st.write("#### Bag of Words")
    # BoW , Tfidf , Cvtz ...

    st.write("#### GRU")
    # BoW , Tfidf , Cvtz ...

    st.write("#### Résultats")

if page == pages[4]:
    st.write("## Analyse de Sentiment")
    # Librairies pré-entraînées : Vader, TextBlob, Bert
    # Mécanisme : Un texte 🡺 un score de tonalité négatif /positif
    # PieChart : Distribution du score de sentiments
    # Diagramme Barre => neutralité générale
    # Graphique : Neutralité des réponses fournisseurs
    # Graphique : distribution rating versus Tonalités sentiments

if page == pages[5]:
    st.write("## Topic Modeling")
    # objectifs : topics commentaires
    # Démarche / LDA
    # Résultats

if page == pages[6]:
    st.write("## Prédiction de la Réponse Fournisseur")
    # objectifs
    # Démarche
    # Résultats

if page == pages[7]:
    st.write("## Démo")
    # commentaires pos, neg, ambigu
    # prédiction du rating
    # prédiction du sentiment
    # prédiction du topic
    # prédiction réponse fournisseur

if page == pages[8]:
    st.write("## Conclusion & Perspectives")
    # Conclusions
    # Perspectives
