import streamlit as st

import pandas as pd
import io

# fichier des avis WebScraps de la Redoute
df = pd.read_csv("redoute.csv", sep=";")


# Titre principal
st.title("Customers Reviews Analytics")

# Barre latérale avec le sommaire
st.sidebar.title("Sommaire")
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
    st.write("## Collecte des Données")
    st.markdown("- Source de données: TrustedShops")
    st.markdown("- Entreprise ciblée : La Redoute")
    st.markdown("- WebScraping avec BeautifulSoup")
    st.markdown("- Données pour l'analyse exploratoire")

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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scikit_posthocs as sp
    import statsmodels.api as sm
    import streamlit as st
    import pandas as pd
    import joblib
    import re
    import nltk
    import string
    import spacy
    import os


    from scipy.stats import levene
    from statsmodels.formula.api import ols
    from scipy.stats import kruskal
    from wordcloud import WordCloud
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from utils import nettoyer_texte
    from utils import extraire_noms
    from collections import Counter
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk import bigrams, trigrams, FreqDist
    from sklearn.cluster import KMeans
    from textblob import TextBlob
    from transformers import pipeline
    from utils import sentiment_score_bert
    from utils import local_css

    nlp = spacy.load('fr_core_news_sm')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')


    # Fonction pour insérer du CSS dans Streamlit
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Insérer le CSS personnalisé
    local_css("style.css")


    st.write("# Prédiction de la Réponse Fournisseur")
    # Chargement des données
    st.text("")
    st.write('## Visualisation du DataFrame')
    df = pd.read_csv('redoute_v31.csv')

    st.text("")
    st.write(df.head())  # Affiche les premières lignes du DataFrame
    st.text("")
    st.write('## EDA (Exploration de données)')
    st.text("")

    # 'createdAt' et 'SupplierReplyDate' 
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['SupplierReplyDate'] = pd.to_datetime(df['SupplierReplyDate'])

    # Calculs nécessaires pour les graphiques
    df['Delay'] = (df['SupplierReplyDate'] - df['createdAt']).dt.days
    average_delay = df['Delay'].mean()
    ratings_per_month = df.groupby(df['createdAt'].dt.to_period("M"))['rating'].mean()
    supplier_replies_per_month = df.groupby(df['SupplierReplyDate'].dt.to_period("M")).size()

    # Choix de graphique dans la zone principale
    choix_graphique = st.selectbox(
    "Sélectionnez le graphique à afficher",
    ('Nombre de réponses fournisseur par jour', 'Histogramme des délais de réponse', 'Analyse du délai moyen par rating', 'Évaluation et réponses fournisseur par mois' )
    )

    if choix_graphique == 'Nombre de réponses fournisseur par jour':
        st.write("#### Nombre total de réponses fournisseur par jour")
        fig, ax = plt.subplots()
        df.groupby(df['createdAt'].dt.date).size().plot(kind='line', ax=ax)
        plt.ylabel('Nombre de réponses')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif choix_graphique == 'Histogramme des délais de réponse':
        st.write("#### Histogramme des délais de réponse")
        fig, ax = plt.subplots()
        plt.hist(df['Delay'], bins=50, edgecolor="k", alpha=0.7)
        plt.axvline(average_delay, color='red', linestyle='dashed', linewidth=1, label=f'Délai moyen: {average_delay:.2f} jours')
        plt.title('Distribution des délais de réponse des fournisseurs')
        plt.xlabel('Délai (en jours)')
        plt.ylabel('Nombre de réponses')
        plt.legend()
        st.pyplot(fig)

    elif choix_graphique == 'Analyse du délai moyen par rating':
        st.write('#### Analyse du délai moyen par rating')
        
        grouped = df.groupby('rating')['Delay'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(grouped['rating'], grouped['Delay'], color=['red', 'orange', 'yellow', 'green', 'blue'])
        ax.set_title('Délai moyen par Rating')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Délai moyen (en jours)')
        ax.set_xticks(grouped['rating'])
        ax.grid(axis='y')
        st.pyplot(fig)

    elif choix_graphique == 'Évaluation et réponses fournisseur par mois':
        st.write("#### Évaluation moyenne et nombre de réponses fournisseur par mois")
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Mois')
        ax1.set_ylabel('Évaluation moyenne', color=color)
        ax1.plot(ratings_per_month.index.astype('str'), ratings_per_month.values, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  

        color = 'tab:red'
        ax2.set_ylabel('Nombre de réponses fournisseurs', color=color)  
        ax2.plot(supplier_replies_per_month.index.astype('str'), supplier_replies_per_month.values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.title('Évaluation moyenne et nombre de réponses fournisseurs par mois')
        st.pyplot(fig)

    
    st.text("")
    st.write("## Tests statistiques")
    st.text("")

    df = pd.read_csv('redoute_v31.csv')

    option_test = st.selectbox(
    "Choisissez le test statistique à effectuer :",
    ("Homogénéité des variances (Test de Levene)", "Normalité des résidus (QQ-plot)", "Test de Kruskal-Wallis")
    )
    
    # 1/ Homogénéité des variances / Test de Levene
    if option_test == "Homogénéité des variances (Test de Levene)":
        st.write("#### 1. Test d'homogénéité des variances (Test de Levene)")
    
        groups = [df['rating'][df['SupplierReply'] == reply] for reply in df['SupplierReply'].unique()]
        statistic, p_value = levene(*groups)
        st.write(f'Statistique de test: {statistic:.4f}')
        st.write(f'P-value: {p_value:.4g}')  # Affiche la p-value 

    # Vérifie si la p-value est inférieure à votre alpha (par exemple, 0.05)
        if p_value < 0.05:
            st.write("Les variances ne sont pas égales selon le test de Levene.")
        else:
            st.write("Les variances sont égales selon le test de Levene.")
    
    # 2/ Normalité des résidus
    elif option_test == "Normalité des résidus (QQ-plot)":
        st.write("#### 2. Normalité des résidus (Graphiquement avec un QQ-plot)")
    
    # Modèle OLS pour les résidus
        model = ols('rating ~ C(SupplierReply)', data=df).fit()
        residus = model.resid
    
    # QQ-plot
        fig = sm.qqplot(residus, fit=True, line="45")
        st.pyplot(fig)

    # Test de Kruskal-Wallis
    elif option_test == "Test de Kruskal-Wallis":
        st.write("#### 3. Test de Kruskal-Wallis")

        groups = [group['rating'].values for name, group in df.groupby('SupplierReply')]
        stat, p_value = kruskal(*groups)

        st.write(f'Statistique de test: {stat}')
        st.write(f'P-value: {p_value}')

        alpha = 0.05
        if p_value < alpha:
            st.write("On rejette l'hypothèse nulle : il existe des différences significatives entre les groupes.")
        else:
            st.write("On ne peut pas rejeter l'hypothèse nulle : il n'y a pas de preuve de différences significatives entre les groupes.")

    # Fonction mise en cache
    @st.cache_data
    def calculer_test_dunn(notes, categories, methode):
        return sp.posthoc_dunn([notes[categories == k] for k in np.unique(categories)], p_adjust=methode)

    # Chargement des données
    df = pd.read_csv('redoute_v31.csv')
    notes = df['rating'].values
    categories = df['SupplierReply'].values

    # Titre 
    st.write("#### 4. Tests de Dunn avec différents ajustements")

    # Sélection de la méthode d'ajustement
    option_ajustement = st.selectbox(
    "Choisissez la méthode d'ajustement pour le test de Dunn :",
    ("fdr_bh","bonferroni", "holm", )
    )

    

    # Calcul et affichage des résultats
    if st.button("Effectuer le test de Dunn", key="unique_key_dunn_test"):
        p_values = calculer_test_dunn(notes, categories, option_ajustement)
        st.write(f"Résultats du test de Dunn avec ajustement {option_ajustement}:")

        # Création de la heatmap de toutes les p-valeurs
        mask = np.triu(np.ones_like(p_values, dtype=bool))
        plt.figure(figsize=(10, 8))
        sns.heatmap(p_values, mask=mask, cmap='viridis', vmax=0.05)
        plt.title('Heatmap des p-valeurs ajustées')
        st.pyplot(plt)

        # Heatmap des comparaisons significatives
        alpha = 0.05
        significant_comparisons = p_values.where(p_values <= alpha)
        significant_mask = mask & (p_values <= alpha)
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(significant_comparisons, mask=significant_mask, cmap='viridis', vmax=0.05, annot=True)
        plt.title('Comparaisons significatives (p <= 0.05)')
        st.pyplot(plt)

        # Affichage des paires significatives
        paires_significatives = np.where(p_values <= alpha)
        paires_indices = list(zip(paires_significatives[0], paires_significatives[1]))

    st.write("## Retour aux Verbatims")
    st.text("")
    st.markdown("""
    Notre analyse montre que les réponses du service client diffèrent selon le rating : 
    génériques pour des notes élevées et personnalisées pour des notes basses, potentiellement due 
    à la satisfaction ou à des difficultés de contact. Les noms des agents peuvent influer sur l'évaluation du sentiment, 
    ce qui nous pousse à examiner de près les commentaires pour les ratings au-dessus et en dessous de 4.
    """, unsafe_allow_html=True)
    st.text("")
    st.write("## Analyse des Réponses pour les Ratings de 4 ou Plus")
    st.text("")
    st.write("#### **Approche WordCloud**")
    st.text("")
    # Importations nécessaires 
    # ... Votre code de prétraitement et de nettoyage ici ...

    # Appliquer la fonction de nettoyage à la colonne 'SupplierReply'
    df_ratings_4_plus = df[df['rating'] >= 4]
    df_ratings_4_plus['SupplierReply_Cleaned'] = df_ratings_4_plus['SupplierReply'].apply(nettoyer_texte)
    # Génération des différents WordClouds

    # Concaténation de tous les textes nettoyés dans une seule chaîne / Création d'une fonction pour générer un wordcloud 
    texte_concatene = ' '.join(df_ratings_4_plus['SupplierReply_Cleaned'])
    
    def generate_wordcloud(stop_words):
        wordcloud = WordCloud(stopwords=stop_words, width=800, height=800, background_color='white').generate(texte_concatene)
        return wordcloud
    
    # Définition des différents ensembles de stop words
    stop_words_base = ['bonjour', 'merci', 'de', 'nous', 'vous', 'votre', 'pour', 'la', 'le', 'et', 'à', 'nos', 'des',
    'en', 'par', 'davoir', 'dans', 'un', 'une', 'sur', 'avec', 'cette', 'que', 'qui', 'plus',
    'sommes', 'notre', 'sont', 'être', 'ou', 'si', 'ils', 'les', 'comme', 'au', 'avoir', 'ce',
    'cet', 'cette', 'ces', 'mais', 'aussi', 'donc', 'lorsque', 'puis', 'car', 'tous', 'tout',
    'très', 'fait', 'faire', 'sans', 'chez', 'toujours', 'jamais', 'peut', 'peuvent', 'aussi',
    'client', 'service', 'équipe', 'ilham', 'temps', 'partager', 'avis', 'expérience', 'pris', 'prendre']

    stop_words_intermediaire = ['bonjour', 'merci', 'de', 'nous', 'vous', 'votre', 'pour', 'la', 'le', 'et', 'à', 'nos', 'des',
    'en', 'par', 'davoir', 'dans', 'un', 'une', 'sur', 'avec', 'cette', 'que', 'qui', 'plus',
    'sommes', 'notre', 'sont', 'être', 'ou', 'si', 'ils', 'les', 'comme', 'au', 'avoir', 'ce',
    'cet', 'cette', 'ces', 'mais', 'aussi', 'donc', 'lorsque', 'puis', 'car', 'tous', 'tout',
    'très', 'fait', 'faire', 'sans', 'chez', 'toujours', 'jamais', 'peut', 'peuvent', 'aussi',
    'client', 'service', 'équipe', 'ilham', 'temps', 'partager', 'avis', 'expérience', 'pris', 'prendre'
    'à', 'de', 'du', 'la', 'le', 'nous', 'vos', 'votre', 'vous', 'et', 'a', 'des', 'en', 'les',
    'un', 'une', 'ont', 'être', 'est', 'pour', 'qui', 'que', 'dans', 'cette', 'vite', 'tout',
    'toute', 'plus', 'si', 'aussi', 'bien', 'comme', 'sans', 'sur', 'ça', 'ont', 'disposition',
    'témoigner', 'commentaire', 'remarques', 'fidélité', 'adresse', 'rubrique', 'remercions',
    'partages', 'site', 'loubna', 'wassim', 'jamila', 'zineb', 'encouragements', 'positif',
    'resterons', 'rester', 'satisfaites', 'satisfaire', 'heureux', 'plaisir', 'client', 'service',
    'équipe', 'ilham', 'temps', 'partager', 'avis', 'expérience', 'pris', 'prendre']  # Ajoutez vos stop words intermédiaires ici

    stop_words_avance = stop_words_intermediaire + ['footer', 'désagrément', 'êtes', 'satisfaite', 'voyons', 'partagées', 'bientôt',
    'aiden', 'resterons', 'contact', 'équip', 'navi', 'gateur', 'navig', 'ateur',
    'compte', 'url', 'suivante', 'ravi', 'beaucoup', 'remercions', 'excusons',
    'lire', 'dautant', 'joindre', 'copiant', 'collant', 'amélior', 'ée', 'améliore',
    'souhaitons', 'souhaite', 'encore', 'adresse', 'contact', 'nous', 'répondre',
    'rencontrés', 'invitons', 'souhaiterions', 'tenté', 'appeler', 'pu', 'moteur',
    'recherche', 'revenir', 'vers', 'jamila', 'wassim', 'soukaina', 'zineb', 'entière',
    'satisfaction', 'attendre', 'aide', 'prochain', 'prochaine', 'tente', 'souhaite',
    'navi', 'gateur', 'essayer', 'rester', 'aider', 'entier', 'entière', 'satisfaire',
    'rester', 'entière', 'entièrement', 'écout', 'équipe', 'équipes', 'proposition',
    'proposer', 'vite', 'rapide', 'rapidement', 'prochainement', 'tenter', 'essayer',
    'atteindre', 'atteint', 'atteints', 'atteinte', 'atteintes', 'déçu', 'déçue',
    'déçus', 'déçues', 'heureux', 'heureuse', 'heureuses', 'content', 'contente',
    'contents', 'contentes', 'satisfait', 'satisfaits', 'satisfaite', 'satisfaites',
    'insatisfait', 'insatisfaite', 'insatisfaits', 'insatisfaites', 'excuse', 'excuses',
    'excusons', 'excuser', 'désolé', 'désolée', 'désolés', 'désolées', 'regret', 'regrets',
    'regrette', 'regretter', 'regretté', 'regrettée', 'regrettés', 'regrettées', 'souci',
    'soucis', 'préoccupation', 'préoccupations', 'inquiétude', 'inquiétudes', 'problème',
    'problèmes', 'problématique', 'problématiques', 'question', 'questions', 'demande',
    'demandes', 'requête', 'requêtes', 'réponse', 'réponses', 'solution', 'solutions',
    'résolution', 'résolutions', 'solliciter', 'sollicité', 'sollicitée', 'sollicités',
    'sollicitées', 'suggestion', 'suggestions', 'conseil', 'conseils', 'recommandation',
    'recommandations', 'avis', 'opinion', 'opinions', 'feedback', 'feedbacks', 'retour',
    'retours', 'critique', 'critiques', 'évaluation', 'évaluations', 'appréciation',
    'appréciations', 'commentaire', 'commentaires']  

    # Dictionnaire des wordclouds
    wordclouds = {
    'Base': generate_wordcloud(stop_words_base),
    'Intermédiaire': generate_wordcloud(stop_words_intermediaire),
    'Avancé': generate_wordcloud(stop_words_avance),
    }

    # Widget pour sélectionner le wordcloud
    option = st.selectbox('Choisissez le niveau de wordcloud', ('Base', 'Intermédiaire', 'Avancé'))

    # Affichage du wordcloud sélectionné
    st.write(f"WordCloud Niveau : {option}")
    wordcloud = wordclouds[option]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)
    st.text("")
    st.markdown("""
    La génération de ses WordCloud successifs après filtrage par des  stop words  donne du sens à notre postulat de départ sur l'hypothèse d'un **caractère générique des réponses**.
    Mais cela ne suffit pas pour justifier l'assertion. Nous avons donc procedé à d'autres test.
    """, unsafe_allow_html=True)
    st.text("")
    st.write("#### **Tests complémentaires**")
    st.text("")

    # Création des cases à cocher pour chaque test
    test_freq_mots = st.checkbox('Analyse de Fréquence des Mots', value=False)
    test_co_occurrence = st.checkbox('Analyse de Co-occurrence', value=False)
    test_bigrammes_trigrammes = st.checkbox('Analyse des Bigrammes/Trigrammes', value=False)
    test_diversite_lexicale = st.checkbox('Diversité Lexicale', value=False)
    test_clustering_texte = st.checkbox('Classification et Clustering de Texte', value=False)
    test_sentiment_textblob = st.checkbox('Analyse de Sentiment simple "TextBlob"', value=False)

    # Condition pour afficher le résultat de l'Analyse de Fréquence des Mots si la case est cochée
    if test_freq_mots:
        # Tokenisation du texte
        mots = texte_concatene.split()

        # Comptage de la fréquence de chaque mot
        frequence_mots = Counter(mots)

        # Convertion du compteur en DataFrame pour une meilleure lisibilité
        df_frequence_mots = pd.DataFrame(frequence_mots.items(), columns=['Mot', 'Fréquence']).sort_values(by='Fréquence', ascending=False)
    
        # Afficher les mots les plus fréquents dans Streamlit
        st.dataframe(df_frequence_mots.head(20))

        # Afficher un histogramme des mots les plus fréquents
        fig, ax = plt.subplots()
        ax.bar(df_frequence_mots['Mot'].head(20), df_frequence_mots['Fréquence'].head(20))
        ax.set_xticklabels(df_frequence_mots['Mot'].head(20), rotation=90)
        ax.set_title("Top 20 des mots les plus fréquents")
        ax.set_xlabel("Mots")
        ax.set_ylabel("Fréquence")

        # Utiliser st.pyplot() pour afficher la figure
        st.pyplot(fig)

        st.write("Résultat de l'Analyse de Fréquence des Mots")

    # Condition pour afficher le résultat de l'Analyse de Co-occurrence si la case est cochée
    if test_co_occurrence:
        # Utiliser CountVectorizer pour créer une matrice de termes
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_ratings_4_plus['SupplierReply_Cleaned'])

        # Conversion de la matrice de termes en array numpy
        Xc = (X.T * X)  # Ceci est la matrice de co-occurrence
        Xc.setdiag(0)  # Remplacer la diagonale par 0s

        # Créer un DataFrame à partir de la matrice de co-occurrence
        noms_mots = vectorizer.get_feature_names_out()
        df_co_occurrence = pd.DataFrame(data=Xc.toarray(), index=noms_mots, columns=noms_mots)

        # Créer un heatmap avec seaborn
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df_co_occurrence.iloc[:20, :20], ax=ax)

        # Afficher le heatmap
        st.pyplot(fig)
        # Afficher la matrice de co-occurrence pour les 20 premiers mots
        st.dataframe(df_co_occurrence.iloc[:20, :20])

        st.write("Résultat de l'Analyse de Co-occurrence")

        

    # Condition pour afficher le résultat de l'Analyse des Bigrammes/Trigrammes si la case est cochée
    if test_bigrammes_trigrammes:
        # Tokenisation du texte
        mots = texte_concatene.split()

        # Créer des listes de bigrammes et trigrammes
        bigramme_liste = list(bigrams(mots))
        trigramme_liste = list(trigrams(mots))

        # Calculer la fréquence des bigrammes et trigrammes
        bigramme_freq = FreqDist(bigramme_liste)
        trigramme_freq = FreqDist(trigramme_liste)

        # Convertir les fréquences en DataFrames
        df_bigrammes = pd.DataFrame(bigramme_freq.most_common(20), columns=['Bigramme', 'Fréquence'])
        df_trigrammes = pd.DataFrame(trigramme_freq.most_common(20), columns=['Trigramme', 'Fréquence'])

        # Afficher les DataFrames
        st.write("Bigrammes les plus courants :")
        st.table(df_bigrammes)

        st.write("Trigrammes les plus courants :")
        st.table(df_trigrammes)

    # Condition pour afficher le résultat de la Diversité Lexicale si la case est cochée
    if test_diversite_lexicale:
        # Tokenisation du texte
        mots = texte_concatene.split()
        # Nombre total de mots
        total_mots = len(mots)

        # Nombre de mots uniques
        mots_uniques = len(set(mots))

        # Calcul de la diversité lexicale
        diversite_lexicale = mots_uniques / total_mots

        # Affichage de la diversité lexicale dans l'application
        st.write(f"Diversité Lexicale: {diversite_lexicale:.2f}")

    # Condition pour afficher le résultat de la Classification et Clustering de Texte si la case est cochée
    if test_clustering_texte:
        # Transformer les données textuelles en TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_avance)
        tfidf = tfidf_vectorizer.fit_transform(df_ratings_4_plus['SupplierReply_Cleaned'])

        # Définir le modèle K-Means
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(tfidf)

        # Afficher les centres de cluster
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = tfidf_vectorizer.get_feature_names_out()

        # Affichage des mots de chaque cluster dans l'application
        st.write("### Résultats de la Classification par K-Means")
        for i in range(5):
            top_ten_words = [terms[ind] for ind in order_centroids[i, :10]]
            cluster_words = ", ".join(top_ten_words)
            st.write(f"Cluster {i}: {cluster_words}")


    # Condition pour afficher le résultat de l'Analyse de Sentiment simple "TextBlob" si la case est cochée
    if test_sentiment_textblob:
        # Ajouter une colonne pour le sentiment
        df_ratings_4_plus['Polarity'] = df_ratings_4_plus['SupplierReply_Cleaned'].apply(lambda text: TextBlob(text).sentiment.polarity)
    
        # Visualiser la distribution du sentiment
        st.write("### Résultat de l'Analyse de Sentiment simple 'TextBlob'")
        fig, ax = plt.subplots()
        sns.histplot(df_ratings_4_plus['Polarity'], bins=30, kde=False, ax=ax)
        ax.set_title('Distribution de Polarité des Sentiments')
        ax.set_xlabel('Polarité')
        ax.set_ylabel('Fréquence')
        st.pyplot(fig)

    st.text("")
    st.write("## Analyse des Réponses pour les Ratings inferieur ou égale à 3")
    st.text("")
    

    # Définir les options de la selectbox

    options_analyse = {
    "Analyse de Contenu": None,
    "Identification du Personnel Répondant": extraire_noms,
    "Séparation des réponses d'Ilham et cumul par intervenant": None  # Pas de fonction associée pour l'instant
    }

    # Widget pour sélectionner l'analyse
    option_selectionnee = st.selectbox(
    "Choisissez l'analyse à afficher :",
    options=list(options_analyse.keys())
    )
    


    # Appliquer l'analyse sélectionnée
    if option_selectionnee == "Analyse de Contenu":
        df_ratings_1_3 = df[df['rating'] <= 3]
        df_ratings_1_3['SupplierReply_Cleaned'] = df_ratings_1_3['SupplierReply'].apply(nettoyer_texte)
    
        # Définition de stop_words
        stop_words = [
        'bonjour', 'merci', 'de', 'nous', 'vous', 'votre', 'pour', 'la', 'le', 'et', 'à', 'nos', 'des',
        'en', 'par', 'davoir', 'dans', 'un', 'une', 'sur', 'avec', 'cette', 'que', 'qui', 'plus',
        'sommes', 'notre', 'sont', 'être', 'ou', 'si', 'ils', 'les', 'comme', 'au', 'avoir', 'ce',
        'cet', 'cette', 'ces', 'mais', 'aussi', 'donc', 'lorsque', 'puis', 'car', 'tous', 'tout',
        'très', 'fait', 'faire', 'sans', 'chez', 'toujours', 'jamais', 'peut', 'peuvent', 'aussi',
        'client', 'service', 'équipe', 'ilham', 'temps', 'partager', 'avis', 'expérience', 'pris', 'prendre'
        'à', 'de', 'du', 'la', 'le', 'nous', 'vos', 'votre', 'vous', 'et', 'a', 'des', 'en', 'les',
        'un', 'une', 'ont', 'être', 'est', 'pour', 'qui', 'que', 'dans', 'cette', 'vite', 'tout',
        'toute', 'plus', 'si', 'aussi', 'bien', 'comme', 'sans', 'sur', 'ça', 'ont', 'disposition',
        'témoigner', 'commentaire', 'remarques', 'fidélité', 'adresse', 'rubrique', 'remercions',
        'partages', 'site', 'loubna', 'wassim', 'jamila', 'zineb', 'encouragements', 'positif',
        'resterons', 'rester', 'satisfaites', 'satisfaire', 'heureux', 'plaisir', 'client', 'service',
        'équipe', 'ilham', 'temps', 'partager', 'avis', 'expérience', 'pris', 'prendre', 'footer', 'désagrément', 'êtes', 'satisfaite', 'voyons', 'partagées', 'bientôt',
        'aiden', 'resterons', 'contact', 'équip', 'navi', 'gateur', 'navig', 'ateur',
        'compte', 'url', 'suivante', 'ravi', 'beaucoup', 'remercions', 'excusons',
        'lire', 'dautant', 'joindre', 'copiant', 'collant', 'amélior', 'ée', 'améliore',
        'souhaitons', 'souhaite', 'encore', 'adresse', 'contact', 'nous', 'répondre',
        'rencontrés', 'invitons', 'souhaiterions', 'tenté', 'appeler', 'pu', 'moteur',
        'recherche', 'revenir', 'vers', 'jamila', 'wassim', 'soukaina', 'zineb', 'entière',
        'satisfaction', 'attendre', 'aide', 'prochain', 'prochaine', 'tente', 'souhaite',
        'navi', 'gateur', 'essayer', 'rester', 'aider', 'entier', 'entière', 'satisfaire',
        'rester', 'entière', 'entièrement', 'écout', 'équipe', 'équipes', 'proposition',
        'proposer', 'vite', 'rapide', 'rapidement', 'prochainement', 'tenter', 'essayer',
        'atteindre', 'atteint', 'atteints', 'atteinte', 'atteintes', 'déçu', 'déçue',
        'déçus', 'déçues', 'heureux', 'heureuse', 'heureuses', 'content', 'contente',
        'contents', 'contentes', 'satisfait', 'satisfaits', 'satisfaite', 'satisfaites',
        'insatisfait', 'insatisfaite', 'insatisfaits', 'insatisfaites', 'excuse', 'excuses',
        'excusons', 'excuser', 'désolé', 'désolée', 'désolés', 'désolées', 'regret', 'regrets',
        'regrette', 'regretter', 'regretté', 'regrettée', 'regrettés', 'regrettées', 'souci',
        'soucis', 'préoccupation', 'préoccupations', 'inquiétude', 'inquiétudes', 'problème',
        'problèmes', 'problématique', 'problématiques', 'question', 'questions', 'demande',
        'demandes', 'requête', 'requêtes', 'réponse', 'réponses', 'solution', 'solutions',
        'résolution', 'résolutions', 'solliciter', 'sollicité', 'sollicitée', 'sollicités',
        'sollicitées', 'suggestion', 'suggestions', 'conseil', 'conseils', 'recommandation',
        'recommandations', 'avis', 'opinion', 'opinions', 'feedback', 'feedbacks', 'retour',
        'retours', 'critique', 'critiques', 'évaluation', 'évaluations', 'appréciation',
        'appréciations', 'commentaire', 'commentaires'
        ] 

        # Initialisation du Vectorizer
        vectorizer = CountVectorizer(stop_words=stop_words)

        # Application du vectorizer aux réponses
        X = vectorizer.fit_transform(df_ratings_1_3['SupplierReply_Cleaned'])

        # Conversion en DataFrame pour une meilleure lisibilité
        df_mots_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

        # Affichage des termes les plus fréquents
        df_mots_freq_sum = df_mots_freq.sum().sort_values(ascending=False).head(20)
    
        # Utilisation de Streamlit pour afficher les résultats
        st.write("Résultats de l'Analyse de Contenu")
        st.dataframe(df_mots_freq_sum)
        pass 

    elif option_selectionnee == "Identification du Personnel Répondant":

        df_ratings_1_3 = df[df['rating'] <= 3].copy()
        df_ratings_1_3['Noms'] = df_ratings_1_3['SupplierReply'].apply(extraire_noms)

        # Afficher les résultats dans Streamlit
        st.write("Résultats de l'Identification du Personnel Répondant :")
        st.dataframe(df_ratings_1_3[['SupplierReply', 'Noms']].head(15))
        
        
    elif option_selectionnee == "Séparation des réponses d'Ilham et cumul par intervenant":

        df_ratings_1_3 = df[df['rating'] <= 3].copy()
        df_ratings_1_3['Noms'] = df_ratings_1_3['SupplierReply'].apply(extraire_noms)

        # Séparer les réponses d'Ilham
        df_ilham = df_ratings_1_3[df_ratings_1_3['Noms'].apply(lambda x: 'Ilham' in x)]

        # Séparer les réponses des autres intervenants
        df_autres = df_ratings_1_3[df_ratings_1_3['Noms'].apply(lambda x: 'Ilham' not in x)]

        # Faire le cumul des réponses par intervenant
        cumul_intervenants = df_ratings_1_3.explode('Noms')['Noms'].value_counts()

        # Afficher les résultats dans Streamlit
        st.write("Réponses d'Ilham :")
        st.dataframe(df_ilham)

        st.write("Réponses des autres intervenants :")
        st.dataframe(df_autres)

        st.write("Cumul des réponses par intervenant :")
        st.dataframe(cumul_intervenants.reset_index().rename(columns={'index': 'Intervenant', 'Noms': 'Nombre de réponses'}))

    

# Chargement et préparation des données
    @st.cache_data
    def charger_donnees(file):
        return pd.read_csv(file, sep=",")

    @st.cache_data
    def preprocess_text(text):
    # Création du stemmer à l'intérieur de la fonction
        stemmer = nltk.SnowballStemmer("french")
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'bonjour,','', text)
        text = re.sub(r'[\w\.-]+ de l\'équipe service client', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('french')]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

    @st.cache_data
    def entrainer_modele_naif(_X_train, _y_train):
        model = LogisticRegression(max_iter=1000)
        model.fit(_X_train, _y_train)
        return model

    @st.cache_data
    def analyser_sentiments(text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)

    # Chargez les données
    df = charger_donnees('redoute_v31.csv')

    # Streamlit UI
    st.text("")
    st.title("Analyse Sentiments / Modélisation VADER, Multi-Classes")
    st.text("")

    # Modélisation Naïve
    st.text("")
    st.header("Modélisation Naïve")
    st.text("")
    df['processed_comments'] = df['SupplierReply'].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_comments'])
    y = df['rating'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_naif = entrainer_modele_naif(X_train, y_train)
    y_pred = model_naif.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Précision du modèle naïf: {accuracy:.2f}")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_comments'])
    joblib.dump(vectorizer, 'vectorizer.joblib')
    

    # Modèle VADER
    st.text("")
    st.header("Modèle VADER")
    st.text("")

    df['sentiment_scores'] = df['SupplierReply'].apply(lambda x: analyser_sentiments(x)['compound'])
    df[['id','SupplierReply', 'sentiment_scores']].to_csv('sentiments_vader.csv', index=False)


    # Modele BERT
    
    # Modele BERT
    st.text("")
    st.text("")
    st.header("Modele BERT")
    st.text("")

    # Chemin du fichier CSV
    chemin_csv = 'sentiments_bert.csv'

    # Vérifier si le fichier CSV existe déjà
    if os.path.isfile(chemin_csv):
        # Charger les scores de sentiment BERT directement depuis le fichier CSV
        df_sentiment_bert = pd.read_csv(chemin_csv)
    else:
        # Charger le pipeline 'sentiment-analysis' avec le modèle multilingue
        nlp_sentiment = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    
        # Fonction pour obtenir le score de sentiment avec BERT
        def sentiment_score_bert(texte):
            try:
                result = nlp_sentiment(texte[:512])[0]  # Tronquer le texte à 512 tokens si nécessaire
                return {'label': result['label'], 'score': result['score']}
            except Exception as e:
                return {'label': 'NEUTRAL', 'score': 0.0}

        # Appliquer la fonction aux réponses pour calculer les scores de sentiment BERT
        df['sentiment_bert'] = df['SupplierReply'].apply(sentiment_score_bert)

        # Sauvegarder les résultats dans un fichier CSV
        df[['id','SupplierReply', 'sentiment_bert']].to_csv(chemin_csv, index=False)

    st.write("Les modèles ont été entraînés et les résultats sont sauvegardés pour la démonstration.")
    
    st.text("")
    st.text("")
    st.text("")

    st.header("**Synthese:**")
    st.text("")
    st.markdown("""
                
    Les réponses des fournisseurs aux avis de 1 à 3 étoiles sont typiques et semi-personnalisées, 
    contrastant avec les réponses plus positives aux avis de 4 étoiles et plus. Cette uniformité, 
    même dans les tentatives de personnalisation comme l'ajout de noms d'employés, indique un script 
    standardisé pour maintenir la constance du service. La standardisation des réponses, évidente dans 
    l'analyse des 'topics', suggère la possibilité d'automatiser les réponses par labels, bien que la détection 
    de sentiment dans les réponses actuelles ne soit pas un indicateur fiable du rating.

    """, unsafe_allow_html=True)

if page == pages[7]:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import re
    import joblib
    from nltk.stem.snowball import SnowballStemmer
    from utils import preprocess_text
    from utils import analyser_sentiments
    from utils import sentiment_score_bert
    from utils import local_css

    # Fonction pour insérer du CSS dans Streamlit
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Insérer le CSS personnalisé
    local_css("style.css")

    st.text("")
    st.text("")
    # commentaires pos, neg, ambigu
    # prédiction du rating
    # prédiction du sentiment
    # prédiction du topic
    # prédiction réponse fournisseur
    st.title("Démonstration des Modèles")
    st.text("")
    st.text("")
    # Chargement du modèle pré-entraîné et du vectoriseur
    model_naif = joblib.load('modele_naif.joblib')
      
    

    # Assurez-vous que le chemin est correct et que 'vectorizer.joblib' existe dans ce chemin
    vectorizer = joblib.load('vectorizer.joblib')
    # Option pour sélectionner le modèle à démontrer
    modele_option = st.selectbox("Choisir le modèle à démontrer:", ["Modèle Naïf", "Modèle VADER","Modèle BERT"])

    if modele_option == "Modèle Naïf":
        # UI pour entrée utilisateur pour le modèle Naïf
        user_input = st.text_area("Entrez un texte pour classification par le modèle Naïf:", "Ce produit est pas mal !")
        if st.button("Classer avec le modèle Naïf"):
            processed_input = preprocess_text(user_input)  
            prediction = model_naif.predict(vectorizer.transform([processed_input]))
            st.write(f"Prédiction de la classe : {prediction[0]}")
 

    elif modele_option == "Modèle VADER":
        df_sentiments = pd.read_csv('sentiments_vader.csv')
        user_selection = st.radio("Choisir un exemple ou entrer un nouveau texte pour VADER", ['Choisir un exemple', 'Entrer un texte'])

        if user_selection == 'Choisir un exemple':
            example_id = st.selectbox("Choisir un ID de commentaire pour VADER:", df_sentiments['id'].values)
            selected_comment = df_sentiments.loc[df_sentiments['id'] == example_id, 'SupplierReply'].iloc[0]
            st.write(f"Commentaire pour l'ID {example_id}: {selected_comment}")
            selected_score = df_sentiments.loc[df_sentiments['id'] == example_id, 'sentiment_scores'].iloc[0]
            st.write(f"Scores de sentiment pour l'ID {example_id}: {selected_score}")
        else:
            user_input = st.text_area("Entrez un texte pour analyse de sentiment par VADER:", "Ce produit est excellent !")
            if st.button("Analyser Sentiment avec VADER"):
                sentiment_scores = analyser_sentiments(user_input)
                st.write(sentiment_scores)
    
    
    elif modele_option == "Modèle BERT":
        df_sentiments_bert = pd.read_csv('sentiments_bert.csv')  # Charger les scores de sentiment BERT
        user_selection = st.radio("Choisir un exemple ou entrer un nouveau texte pour BERT", ['Choisir un exemple', 'Entrer un texte'])

        if user_selection == 'Choisir un exemple':
            example_id = st.selectbox("Choisir un ID de commentaire pour BERT:", df_sentiments_bert['id'].values)
            selected_comment = df_sentiments_bert.loc[df_sentiments_bert['id'] == example_id, 'SupplierReply'].iloc[0]
            st.write(f"Commentaire pour l'ID {example_id}: {selected_comment}")
            selected_score = df_sentiments_bert.loc[df_sentiments_bert['id'] == example_id, 'sentiment_bert'].iloc[0]
            st.write(f"Scores de sentiment BERT pour l'ID {example_id}: {selected_score}")
        else:
            user_input = st.text_area("Entrez un texte pour analyse de sentiment par BERT ou choisissez un exemple ci-dessous:", "J'ai commandé un colis mais il n'est jamais arrivé.")

            exemples_textes = [
                "J'ai commandé un colis mais il n'est jamais arrivé.",
                "La livraison a été retardée sans aucune notification préalable.",
                "Mon paquet est arrivé endommagé, très déçu du service.",
                "Le suivi de mon colis indique qu'il a été livré mais je ne l'ai pas reçu.",
                "Très mécontent, mon produit est arrivé trois semaines en retard.",
                "Le service client ne m'a pas aidé à résoudre mon problème de livraison.",
                "J'ai reçu le mauvais produit et je dois maintenant attendre le retour.",
                "Commande perdue par le transporteur, et maintenant je dois attendre un remboursement.",
                "L'article livré ne correspond pas à la description du site, problèmes de retour.",
                "Pas de communication de la part du transporteur, je ne sais pas où est mon colis."
            ]

            exemple_selectionne = st.selectbox("Ou sélectionnez un exemple de problème de livraison :", exemples_textes)

            if st.button("Analyser Sentiment avec BERT"):
                # S'il y a une saisie de l'utilisateur, utilisez-la, sinon utilisez l'exemple sélectionné
                texte_a_analyser = user_input if user_input else exemple_selectionne
                sentiment_result = sentiment_score_bert(texte_a_analyser)
                st.write(sentiment_result)







    # elif modele_option == "Modèle BERT":
    #     df_sentiments_bert = pd.read_csv('sentiments_bert.csv')  # Charger les scores de sentiment BERT
    #     user_selection = st.radio("Choisir un exemple ou entrer un nouveau texte pour BERT", ['Choisir un exemple', 'Entrer un texte'])

    #     if user_selection == 'Choisir un exemple':
    #         example_id = st.selectbox("Choisir un ID de commentaire pour BERT:", df_sentiments_bert['id'].values)
    #         selected_comment = df_sentiments_bert.loc[df_sentiments_bert['id'] == example_id, 'SupplierReply'].iloc[0]
    #         st.write(f"Commentaire pour l'ID {example_id}: {selected_comment}")
    #         selected_score = df_sentiments_bert.loc[df_sentiments_bert['id'] == example_id, 'sentiment_bert'].iloc[0]
    #         st.write(f"Scores de sentiment BERT pour l'ID {example_id}: {selected_score}")
    #     else:
    #         user_input = st.text_area("Entrez un texte pour analyse de sentiment par BERT:", "Ce produit est excellent !")
    #         if st.button("Analyser Sentiment avec BERT"):
    #             sentiment_result = sentiment_score_bert(user_input)
    #             st.write(sentiment_result)



if page == pages[8]:
    st.write("## Conclusion & Perspectives")
    # Conclusions
    # Perspectives
