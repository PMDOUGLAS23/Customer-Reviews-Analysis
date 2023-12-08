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
    

    st.write("## Prédiction de la Réponse Fournisseur")
    # Chargement des données
    st.write('### Visualisation du DataFrame')
    df = pd.read_csv('redoute_v31.csv')
    st.write(df.head())  # Affiche les premières lignes du DataFrame

    st.write('### EDA (Exploration de données)')
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
        st.write("### Nombre total de réponses fournisseur par jour")
        fig, ax = plt.subplots()
        df.groupby(df['createdAt'].dt.date).size().plot(kind='line', ax=ax)
        plt.ylabel('Nombre de réponses')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif choix_graphique == 'Histogramme des délais de réponse':
        st.write("### Histogramme des délais de réponse")
        fig, ax = plt.subplots()
        plt.hist(df['Delay'], bins=50, edgecolor="k", alpha=0.7)
        plt.axvline(average_delay, color='red', linestyle='dashed', linewidth=1, label=f'Délai moyen: {average_delay:.2f} jours')
        plt.title('Distribution des délais de réponse des fournisseurs')
        plt.xlabel('Délai (en jours)')
        plt.ylabel('Nombre de réponses')
        plt.legend()
        st.pyplot(fig)

    elif choix_graphique == 'Analyse du délai moyen par rating':
        st.write('### Analyse du délai moyen par rating')
        
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
        st.write("### Évaluation moyenne et nombre de réponses fournisseur par mois")
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

    import scikit_posthocs as sp
    import statsmodels.api as sm
    from scipy.stats import levene
    from statsmodels.formula.api import ols
    from scipy.stats import kruskal
    import streamlit as st
    import pandas as pd
    import numpy as np
    import scikit_posthocs as sp
    
    st.write("## Tests statistiques")
    df = pd.read_csv('redoute_v31.csv')

    option_test = st.selectbox(
    "Choisissez le test statistique à effectuer :",
    ("Homogénéité des variances (Test de Levene)", "Normalité des résidus (QQ-plot)", "Test de Kruskal-Wallis")
    )
    
    # 1/ Homogénéité des variances / Test de Levene
    if option_test == "Homogénéité des variances (Test de Levene)":
        st.write("### 1. Test d'homogénéité des variances (Test de Levene)")
    
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
        st.write("### 2. Normalité des résidus (Graphiquement avec un QQ-plot)")
    
    # Modèle OLS pour les résidus
        model = ols('rating ~ C(SupplierReply)', data=df).fit()
        residus = model.resid
    
    # QQ-plot
        fig = sm.qqplot(residus, fit=True, line="45")
        st.pyplot(fig)

    # Test de Kruskal-Wallis
    elif option_test == "Test de Kruskal-Wallis":
        st.write("### 3. Test de Kruskal-Wallis")

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
    st.write("## 4. Tests de Dunn avec différents ajustements")

    # Sélection de la méthode d'ajustement
    option_ajustement = st.selectbox(
    "Choisissez la méthode d'ajustement pour le test de Dunn :",
    ("bonferroni", "holm", "fdr_bh")
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
