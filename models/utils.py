
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import spacy
from transformers import pipeline

# Charger le modèle français
nlp = spacy.load('fr_core_news_sm')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# Fonction pour insérer du CSS dans Streamlit
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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


def analyser_sentiments(text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)


def nettoyer_texte(texte):

    texte_sans_ponctuation = texte.translate(str.maketrans('', '', string.punctuation))

    # Convertir le texte en minuscules
    texte_minuscules = texte_sans_ponctuation.lower()

    # Enlever les chiffres (si nécessaire)
    texte_sans_chiffres = ''.join([i for i in texte_minuscules if not i.isdigit()])

    return texte_sans_chiffres

def extraire_noms(texte):
    doc = nlp(texte)
    return [ent.text for ent in doc.ents if ent.label_ == 'PER']

nlp_sentiment = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')


def sentiment_score_bert(texte):
    try:
        result = nlp_sentiment(texte[:512])[0]  # Tronquer le texte à 512 tokens si nécessaire
        return {'label': result['label'], 'score': result['score']}
    except Exception as e:
        return {'label': 'NEUTRAL', 'score': 0.0}

