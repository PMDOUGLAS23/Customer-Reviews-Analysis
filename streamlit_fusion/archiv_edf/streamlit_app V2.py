import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.write(
        "Il est important -  avant de conduire ce type d’analyse - **de préparer le texte** afin qu’il soit le plus exploitable possible aux différents outils disponibles."
    )
    st.write(
        "On parle de pre-processing notamment en utilisant des techniques de **racinisation, tokenisation, utilisation d'expressions regulieres et suppression des stop words**."
    )

    import streamlit as st

# Titre
st.header("Résumé de l'Analyse de Sentiments")

# Introduction
st.write(
    "On constate à travers nos diverses itérations que l’analyse de texte peut s’avérer ardue et reste  un processus très itératif. Il en ressort aussi que certains outils vont très bien identifier la neutralité d’un commentaire mais par contre ne vont pas supporter une bonne prédiction de note.."
)

# Approches non supervisées
st.header("Approches non supervisées :")

# Word Cloud
st.subheader("Word Cloud :")
st.write(
    "Création de 'Word Cloud' à partir des commentaires pour visualiser les thèmes dominants.  Il s’agit d’un **outil très pratique**, qui combiné à d’autres techniques comme la vectorisation peut se révéler intéressant."
)

# Sentiment Analysis
st.subheader("Outils à disposition :")
st.write(
    "L'analyse de sentiments couvre diverses méthodes, de la simple utilisation de lexiques à des approches avancées avec transformers. Nous avons utilisé des outils tels que **Word Cloud, Text Blob, Vader, Naive Bayes et Bert**. **Multinomial Naive Bayes** est un algorithme de classification bayésienne naïve qui est adapté pour des données catégorielles discrètes. **BERT**(Bidirectional Encoder Representations from Transformers) est un modèle de traitement du langage naturel basé sur des transformers, qui sont des modèles de réseaux de neurones particulièrement performants dans le traitement séquentiel de données."
)

# Exemple avec Vader
st.subheader("Prenons un exemple avec Vader :")
st.write(
    "VADER utilise un lexique pré-étiqueté qui prend en compte la polarité, l'intensité et les émotions liées aux mots spécialement conçu pour gérer les nuances du langage naturel."
)
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("redoute_v3.csv")

# Téléchargement du lexique VADER pour l'analyse de sentiment
nltk.download("vader_lexicon")

# Sélection d'un exemple de commentaire
comment = st.text_area(
    "Exemple:",
    "je commande depuis longtemps chez la redoute; j'y apprécie le choix , la rapidité de livraison et la facilité de règlement grâce à la carte ,même si presque  à chaque fois je demandais un règlement comptant cela m'évitait de laisser mon numéro de carte bleue  sur le web ...par contre depuis quelques jours cela a changé je n'y comprends plus rien je ne peux plus payer comme ça alors que  je n'ai absolument  aucun débit  en attente sur mon compte ....du coup par sécurité face à l'internet  j'ai dû régler  mes deux dernières commandes avec une carte bancaire à usage unique ce qui est un peu  plus compliqué pour moi et risque  forcement de me faire hésiter pour des   futurs  achats sur ce site....dommage.",
)

# Initialisation de l'analyseur de sentiment VADER
sid = SentimentIntensityAnalyzer()

# Calcul du score de sentiment
sentiment_score = sid.polarity_scores(comment)

# Vader result on this specific example
x = [0.0, 0.977, 0.023, 0.4215]

st.set_option("deprecation.showPyplotGlobalUse", False)
plt.figure(figsize=(8, 6))
plt.pie(x, labels=["negative", "neutral", "positive", "coumpound"])
plt.title("Distribution of VADER Sentiment Scores")
plt.legend()
st.pyplot()

st.write(
    " Le resultat pour cet example exprime plus de **97% de neutralité**, en effet la lecture du texte révèle une tonalité mixte avec des aspects positifs et des préoccupations."
)


# Affichage du score de sentiment
# st.write("Score de sentiment :", sentiment_score)


# BERT
st.subheader("BERT:")
st.write(
    "L'utilisation de BERT pré-entraîné pour l'analyse de sentiment est puissante car il peut comprendre le contexte et les relations entre les mots de manière plus sophistiquée que les méthodes traditionnelles. Les resultats sont plus proches du rating client que Vader, améliorant la précision de l'évaluation du sentiment."
)

# st.write(' ## Nous avons utilisé plusieurs librairies dites pré-entraînées telles que Vader, TextBlob, Bert')
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
    st.write(
        """
**Objectifs :** Dans le but de satisfaire notre client et de faciliter la catégorisation des commentaires clients et des réponses, nous avons exploré le topic modeling avec le modèle Latent Dirichlet Allocation de la librairie Gensim en Python.

**Méthodologie :**
- Définition d’un corpus de mots : nos commentaires nettoyés et tokenisés, rassemblés en bag of words.
- Application du modèle LDA et création d’un dictionnaire établissant la correspondance entre les mots du corpus et les identifiants du modèle.

**Résultats :** L’analyse LDA nous a permis d'identifier les sujets dominants pour chaque commentaire, révélant des thèmes liés à la vente par correspondance, tels que les problèmes de livraison, de paiement, de conformité de taille, etc.
"""
    )

# Exemple de Sentiment Analysis Bert, combiné au Topic modeling via un Chatbot

import pandas as pd
import numpy as np

from transformers import BertTokenizer, TFBertForSequenceClassification

from gensim import corpora, models

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download("punkt")
# nltk.download("stopwords")


df = pd.read_csv("redoute_v3.csv")
df = df.head(20)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download("punkt")
# nltk.download("stopwords")

# word.isalpha() permet de filtrer que les caracteres, va enlever les chiffres, ponctuations et caracteres speciaux


# Fonction pour nettoyer le texte
def nettoyer_texte(comment):
    stop_words = set(stopwords.words("french"))
    ps = PorterStemmer()
    words = [
        ps.stem(word)
        for word in word_tokenize(comment.lower())
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(words)


# Créer une nouvelle colonne 'comment_cleaned' en appliquant la fonction nettoyer_texte
df["comment_cleaned"] = df["comment"].apply(nettoyer_texte)


# nettoyage supplementaire avec une regex qui prend uniquement les mots de 4 caracteres
from nltk.tokenize.regexp import RegexpTokenizer
import pandas as pd


def apply_regex(chaine):
    tokenizer = RegexpTokenizer("[a-zA-Zé]{4,}")
    tokens_regex = tokenizer.tokenize(chaine)
    return " ".join(tokens_regex)  # pour rejoindre les tokens en une chaîne


# Appliquation de la fonction à la colonne 'comment_cleaned'
df["comment_cleaned"] = df["comment_cleaned"].apply(apply_regex)


# Charger le modèle de sentiment BERT
model = TFBertForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)
tokenizer = BertTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)


# Appliquer l'analyse de sentiment BERT à la colonne 'comment_cleaned'
def predict_sentiment(comment):
    tokens = tokenizer.encode_plus(
        comment, return_tensors="tf", max_length=512, truncation=True
    )
    logits = model(tokens)["logits"]
    return np.argmax(logits.numpy())


df["sentiment_bert"] = df["comment_cleaned"].apply(predict_sentiment)


# Créer le dictionnaire et le corpus basés sur les commentaires nettoyés (bag of words)
dictionary = corpora.Dictionary(df["comment_cleaned"].apply(word_tokenize))
corpus = [
    dictionary.doc2bow(word_tokenize(comment)) for comment in df["comment_cleaned"]
]


def topic_modeling(comment, corpus, dictionary, num_topics=5):
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    topics = lda_model.print_topics(num_words=3)
    return topics


# Créer le dictionnaire et le corpus basés sur les commentaires nettoyés
# dictionary = corpora.Dictionary(df['comment_cleaned'].apply(word_tokenize))
# corpus = [dictionary.doc2bow(word_tokenize(comment)) for comment in df['comment_cleaned']]

# Créer le modèle LDA
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Obtenir les termes les plus fréquents pour chaque topic - ici j'ai choisi 4 topics via topn
# on parcourt chaque topic du modele lda et on les stocke ds topic_terms
topic_terms = {}
for topic_id in range(lda_model.num_topics):
    terms = lda_model.show_topic(topic_id, topn=4)
    topic_terms[topic_id] = [term for term, _ in terms]


def get_dominant_topic(comment, lda_model, dictionary):
    # Utiliser le modèle LDA pour obtenir la distribution des topics
    bow_vector = dictionary.doc2bow(word_tokenize(comment))
    topics_distribution = lda_model.get_document_topics(bow_vector)

    # Trouver le topic dominant (numéro de topic) avec la probabilite la plus elevee
    # rappel : topics_distribution:  chaque élément est un tuple de la forme (topic_id, probability)
    dominant_topic = max(topics_distribution, key=lambda x: x[1])[0]

    return dominant_topic


def get_topic_name(topic_id):
    # Obtenir les termes les plus fréquents du topic
    terms = topic_terms.get(topic_id, [])
    return ", ".join(terms)


# Titre de l'application - chatbot
st.subheader("Chatbot interactif avec Analyse de sentiment Bert et topic modeling")

# Section pour choisir un commentaire
selected_comment_index = st.number_input(
    "Saisissez l'index du commentaire :",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1,
)

# Affichage du commentaire sélectionné
selected_comment = df.loc[selected_comment_index, "comment_cleaned"]
st.subheader("Commentaire sélectionné et nettoyé:")
st.write(selected_comment)

# Analyse de sentiment avec BERT
sentiment_label = predict_sentiment(selected_comment)
st.subheader("Analyse de Sentiment BERT :")
st.write(f"Sentiment : {sentiment_label}")
st.write(
    "Pour rappel, l'échelle Bert du modèle choisi va de 0 a 4. 0 étant très négatif et 4 très positif"
)

# Topic Modeling
dominant_topic = get_dominant_topic(selected_comment, lda_model, dictionary)
topic_name = get_topic_name(dominant_topic)
st.subheader("Topic Modeling :")
st.write(f"Topics dominants : {topic_name}")

# Rating client
rating = df.loc[selected_comment_index, "rating"]
st.subheader("Rating client :")
st.write(f"Rating client: {rating}")

print("---")

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
    st.header("Conclusion & Perspectives")
    # Conclusions
    # Perspectives
    # Introduction
    st.markdown(
        "Notre projet visait à prédire les notations des clients, catégoriser les commentaires, proposer des réponses automatiques, et analyser les sentiments."
    )

    # Approche 1 : Modèles de classification
    st.subheader("Modèles de classification")
    st.markdown(
        "L’utilisation de divers modèles pour prédire le sentiment client en se basant sur la longueur du commentaire, du titre, et la durée depuis la transaction. **La régression logistique simple a montré des performances notables**."
    )

    # Approche 2 : Bag of Words avec TF IDF et Count Vectorizer
    st.subheader("Bag of Words avec TF IDF et Count Vectorizer")
    st.markdown(
        "L'application du **Bag of Words avec TF IDF et Count Vectorizer** a permis **une bonne précision d’ensemble**. Cependant, des limitations subsistent dans la précision de prédiction du sentiment négatif."
    )

    # Approche 3 : Réseaux de Neurones Récurrents GRU
    st.subheader("Réseaux de Neurones Récurrents GRU")
    st.markdown(
        "**Les GRU ont surpassé les modèles précédents, atteignant des précisions globales dépassant 91%**, améliorant nettement la précision dans la prédiction du sentiment négatif. Le rééchantillonnage et la prise en compte du poids des classes pour les réseaux de neurones ont été essentiels pour équilibrer les données quel que soit le modèle."
    )

    # Prédiction du retour fournisseur
    st.subheader("Prédiction du retour fournisseur")
    st.markdown(
        "Une approche parallèle a été adoptée, mettant l'accent sur le feature engineering, la recherche d'une trame standard, et le fine-tuning des modèles. L'analyse des commentaires fournisseur a révélé des **tendances spécifiques** telles qu’une **standardisation des réponses**, confirmant nos hypothèses."
    )

    # Analyse des sentiments
    st.subheader("Analyse des sentiments")
    st.markdown(
        "Des word clouds à BERT, les approches supervisées et semi-supervisées, notamment Bert et le Topic Modeling, ont fourni des informations tres tangibles."
    )

    # Chatbot interactif
    st.subheader("Chatbot interactif")
    st.markdown(
        "La mise en place d'un **Chatbot interactif**, intégrant une évaluation des sentiments et présentant les quatre principaux sujets sous-jacents, a rendu **conviviale l'analyse des commentaires**. Les résultats concluants mettent en lumière l'utilité de cet outil."
    )

    # Conclusion
    st.subheader("Conclusion")
    st.markdown(
        "En résumé, nos analyses approfondies nous ont permis d’élaborer un bon système de prédictions de l'évaluation des commentaires, mais comme dans tout modèle de prédiction, des efforts continus sont nécessaires pour le perfectionner et optimiser les paramètres. À un stade avancé, une modélisation de sujets plus précis pourrait diriger les commentaires vers les départements spécifiquement concernés."
    )
