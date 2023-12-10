import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import nltk
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download("punkt")
# nltk.download("stopwords")

from transformers import BertTokenizer, TFBertForSequenceClassification

from gensim import corpora, models


#  fonction pour charger des fichers csv
@st.cache_data  # pour réduire le temps de chargement du dataframe
def load_data(file_name, sep=","):
    return pd.read_csv(file_name, sep=sep, index_col=0)


# Titre principal
st.title(":blue[Customers Reviews Analytics]")

# Barre latérale avec le sommaire
st.sidebar.title(":blue[Sommaire]")
pages = [
    "Introduction",
    "Collecte des Données",
    "Exploration et Analyse des Données",
    "Prédiction supervisée du sentiment",
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
    st.write("## Objectifs du Projet")
    st.markdown("###### - Collecter des avis clients sur des plateformes en ligne")
    st.markdown("###### - Prédire la statisfaction client")
    st.markdown("###### - Catégoriser les commentaires clients")
    st.markdown("###### - Proposer des réponses automatiques aux commentaires")
    st.markdown("###### - Analyser le sentiment des avis des clients")

    st.write("## Equipe Projet")
    st.text(
        """
        Michel Douglas Piamou
        Mike Boudhabhay
        Edwige Fève
        """
    )

if page == pages[1]:
    st.write("## Collecte des Données")
    st.divider()
    # Source de données
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Source de données")
    with col2:
        st.markdown(
            """ **TrustedShops**, Entreprise allemande qui propose entre autres:
                        Certification de sites web marchands, 
                        Services **d'évaluation et d'avis clients**
                    """,
            unsafe_allow_html=True,
        )
    st.divider()
    # Entreprise cible
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Entreprise ciblée")
    with col4:
        st.markdown(
            "**La Redoute**, Leader français du e-commerce en mode et maison",
            unsafe_allow_html=True,
        )
    st.divider()
    col5, col6 = st.columns(2)
    with col5:
        st.markdown(
            "##### WebScraping des données des avis clients", unsafe_allow_html=True
        )
    with col6:
        st.markdown(
            "1 - **requests.get + BeautifulSoup + Pandas**",
            unsafe_allow_html=True,
        )
        st.markdown(
            "2 - 1er nettoyage et formatage des données, puis stockage dans un fichier .csv",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown("##### Données pour l'Analyse Exploratoire", unsafe_allow_html=True)
    df = load_data("redoute.csv", sep=";")
    st.dataframe(df.head(5))
    # pour afficher df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

if page == pages[2]:
    # chargement des données
    df = load_data("redoute_v3.csv")
    # selection des colonnes utiles
    cols = [
        "rating",
        "comment_length",
        "log_comment_length",
        "title_length",
        "nb_days_before_review",
        "month_of_cmt",
        "day_of_cmt",
        "weekday_of_cmt",
        "cmt_in_weekend",
        "hour_of_cmt",
    ]
    df = df[cols]

    st.write("## Exploration et Analyse des Données")
    st.divider()
    st.markdown(" ##### Traitement des valeurs manquantes")
    st.divider()
    st.markdown(" ##### Enrichissement du jeu de données")
    st.text(
        "- Mois, jours ouvrés/week-end, heures auxquels les commentaires ont été postés"
    )
    st.text("- Longeur du commentaire")
    st.text(" - Logarithme de la longeur du commentaire")
    st.text("- Longueur du titre du commentaire")
    st.text(
        "- Nombre de jours entre la date de transaction et la date de création du commentaire "
    )
    st.divider()
    st.markdown("##### Analyse des distributions")

    # Distribution du rating selon la date/temps choisi de l'utilisateur
    options = [
        "",
        "month_of_cmt",
        "weekday_of_cmt",
        "cmt_in_weekend",
        "hour_of_cmt",
    ]
    # menu de sélection des variable pour le Box plot
    var_selected = st.selectbox(
        "###### Afficher le distribution du rating selon le temps/date de votre choix",
        options,
    )
    # Box plot de la longeur des commentaires vs rating
    if var_selected == "":
        fig = plt.figure(figsize=(3, 3))
        sns.countplot(data=df, x="rating", palette="pastel")
        plt.title("Répartition du rating")
    else:
        fig = plt.figure(figsize=(3, 3))
        sns.countplot(data=df, x="rating", hue=var_selected, palette="pastel")
        plt.title("Distribution du rating vs " + var_selected)
    st.pyplot(fig)

    # Affichage du BOXPLOT des variables explicatives

    # Choix d'une variable pour afficher le boxplot
    options = [
        "comment_length",
        "log_comment_length",
        "title_length",
        "nb_days_before_review",
    ]

    # menu de sélection des variable pour le Box plot
    var_selected = st.selectbox(
        "###### Sélectionnez une variable pour afficher sa distribution", options
    )

    # Box plot de la longeur des commentaires vs rating
    fig = plt.figure(figsize=(2, 3))
    sns.boxplot(
        data=df, x="rating", y=var_selected, hue="rating", palette="Set2", legend=False
    )
    plt.title("Box plot - " + var_selected)
    st.pyplot(fig)

    # Affichage heatmap des corrélations
    st.divider()
    st.markdown(" ##### Analyse des corrélations")
    fig = plt.figure(figsize=(7, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    st.pyplot(fig)
    st.divider()
    st.markdown("##### Test statistiques ")
    st.text("- test ANOVA : rating versus log_comment_length")
    st.text(
        " - test de Kruska Wallis : effet du rating sur les  variables explicatives"
    )
    st.text(" - test de spearman : corrélations entre variables explicatives")

    st.divider()

    st.markdown(
        "##### Mesure de la colinéarité entre variables explicatives : calcul du VIF"
    )


if page == pages[3]:
    # Chargement des données des performances
    df_perf_bin = load_data("data_perf.csv")

    # graphique radar des métriques d'une liste de modèles pour clf binaire
    def plot_radar(data, model_list, best_model):
        metrics = data.index.to_list()
        fig = go.Figure()
        for model in models_list:
            fig.add_trace(
                go.Scatterpolar(
                    r=data[model].to_list(),
                    theta=metrics,
                    # fill='toself',
                    name=model,
                )
            )
        fig.update_layout(
            title="Métriques : " + model_selected + " vs " + best_model,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            width=600,
            height=600,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("## Prédiction supervisée du sentiment")

    st.divider()
    st.write("### Classification Binaire")
    st.latex(
        r"sentiment = \begin{cases} 0 & \text{si } rating \leq 3 \\ 1 & \text{si } rating \geq 4 \end{cases}"
    )
    st.divider()
    st.write("### Classification Multi-classes")
    st.latex(
        r"sentiment = \begin{cases} -1 & \text{si } rating < 3 \\ 0 & \text{si } rating = 3 \\1 & \text{si } rating > 3 \end{cases}"
    )

    st.divider()
    st.write("### Démarche")
    st.markdown(
        "- Une modélisation naïve dite Baseline, Approche Bag of Words, Deep Learning"
    )
    st.markdown(
        "- **Sous-échantillonage aléatoire** ou utilisation du poids des classes pour prendre en compte le déséquilibre des classes"
    )

    st.markdown("- Métriques prinicipales de comparaison des modèles :")
    st.text("Accuracy, Precision du sentiment négatif, Rappel du sentiment négatif")
    st.text("F1-macro, Rappel-macro,  Precision-macro")
    st.text("matrices de confusion")

    st.markdown("- Recherche de meilleurs  hyperparmètres avec GridSearchCV")

    st.divider()
    st.write("### 1. Modélisation de Base")
    st.image("st_process_bsl_modeling.jpg")
    st.markdown("Prédiction du sentiment à partir de 3 variables explicatives")

    # Métriques des modèles baseline
    # Options de choix des modèles baseline
    options = [
        "dec_tree_bsl",
        "r_fo_bsl",
        "g_nb_bsl",
        "m_nb_bsl",
        "c_nb_bsl",
        "lsvc_bsl",
        "svc_bsl",
        "knn_bsl",
        "gdb_bsl",
        "grid_log_reg_bsl_grid",
        "grid_lsvc_bsl_grid",
        "grid_svc_bsl_grid",
        "grid_gdb_bsl_grid",
    ]
    # menu de sélection des variable pour le Box plot
    model_selected = st.selectbox(
        "###### Sélectionnez un modèle baseline pour afficher ses métriques", options
    )
    # Radar des métriques du modèle choix versus regression logistique (log_reg_bsl)
    models_list = ["log_reg_bsl", model_selected]
    data = df_perf_bin.iloc[:, :-2].set_index("model").T
    plot_radar(data, models_list, "log_reg_bsl")

    st.divider()
    st.write("### 2. Bag of Words")
    st.image("st_process_bag_of_word.jpg")

    st.markdown("Chaque commentaire est transformé en un **vecteur numérique**")
    st.markdown(
        """**Count Vectorization**: compter le nombre d'occurrences des termes dans le commentaire"""
    )
    st.markdown(
        """**Pondération TF-IDF** tient compte de la fréquence du terme dans le commentaire (TF) et de 
        l'inverse de la fréquence du terme dans l'ensemble des commentaires (IDF)"""
    )

    # menu de sélection des modèles pour comparaison des métriques
    options_bow = df_perf_bin[df_perf_bin["model_cat"].isin(["tfidf", "cvtz"])][
        "model"
    ].to_list()
    options_bow.remove("Logistic_reg_cvz")
    model_selected = st.selectbox(
        "###### Sélectionnez un modèle pour afficher ses métriques", options_bow
    )
    # Radar des métriques du modèle choix versus regression logistique (log_reg_bsl)
    models_list = ["Logistic_reg_cvz", model_selected]
    data = df_perf_bin.iloc[:, :-2].set_index("model").T
    plot_radar(data, models_list, "Logistic_reg_cvz")

    # Réseau de neuronnes Gates Recurrent Units
    # wordcloud pour définition RNN
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Liste de mots clés représentatifs d'un RNN
    rnn_keywords = [
        "sequence",
        "recurrent",
        "neural network",
        "context",
        "LSTM",
        "GRU",
        "backpropagation",
        "embedding",
        "prediction",
        "generation",
        "time series",
        "memory",
        "hidden state",
        "long short-term memory",
        "contextual",
        "sequential",
        "understanding",
        "language model",
        "dependency",
        "stateful",
    ]
    # nuage de mots avec les mots clés RNN
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(rnn_keywords)
    )

    st.divider()
    st.write("### 3. Réseau de Neurones")
    st.image("st_process_neural_network.jpg")

    st.divider()
    st.markdown("#### Réseau de neurones récurrents")
    st.image(wordcloud.to_array(), caption="Nuage de mots représentatif d'un RNN")

    st.divider()
    st.markdown("#### GRU - Gated Recurrent Units")
    st.markdown(
        """ - Variante des réseaux de neurones récurrents qui permettent de capturer 
        les dépendances à long terme dans les séquences de données."""
    )
    st.markdown(
        """ - Alternative au RNN tradidionnel pour palier au problème de disparition du gradient"""
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Architecture d'une cellule d'un GRU :")
        st.markdown("- une porte de réinitialisation (Reset Gate)")
        st.markdown("- une porte d'oubli (Update Gate)")
    with col2:
        st.image("st_gru_cell.jpg")

    # code du modèle GRU_1
    st.markdown("##### Exemple de modèle GRU mis en oeuvre:")
    code_python_GRU = """
        embedding_size = 100
        model = Sequential()
        model.add(Embedding(input_dim=num_words, 
                            output_dim=embedding_size, 
                            input_length=max_tokens,
                            name="embedding_layer"))
        model.add(GRU(units=128, return_sequences=True))
        model.add(GRU(units=64, return_sequences=True)) 
        model.add(GRU(units=12))
        model.add(Dense(1, activation='sigmoid'))
        # Compile
        optimizer = Adam(learning_rate=0.008)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    """
    st.code(code_python_GRU, language="python")

    # ---------------------------------------
    # menu de sélection des modèles pour comparaison des métriques
    options_dl = df_perf_bin[df_perf_bin["model_cat"].isin(["GRU", "LSTM"])][
        "model"
    ].to_list()
    options_dl.remove("GRU_1")
    model_selected = st.selectbox(
        "###### Sélectionnez un modèle pour afficher ses métriques", options_dl
    )
    # Radar des métriques du modèle choix versus modèle GRU_1
    models_list = ["GRU_1", model_selected]
    data = df_perf_bin.iloc[:, :-2].set_index("model").T
    plot_radar(data, models_list, "GRU_1")

    # -----------------------------------

    st.write("### Résultats comparés - Classification Binaire")

    # Accuracy des modèles
    col_metrics = [
        "model",
        "accuracy",
        "recall_0",
        "precision_0",
        "recall_macro",
        "precision_macro",
    ]
    top_models = [
        "log_reg_bsl",
        "Logistic_reg_cvz",
        "SVC(kernel=rbf)_tfidf",
        "GRU_esz50_1",
        "GRU_1",
    ]

    # 1. Precision et Recall du Sentiment Négatif
    fig = px.scatter(
        df_perf_bin,
        x="recall_0",
        y="precision_0",
        size="size",
        color="model_cat",
        hover_name="model",
        hover_data=["model", "accuracy"],
        log_x=False,
        size_max=25,
        title="Précision et Recall du sentiment négatif",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # 2. Accuracy vs Rappel du sentiment Négatif
    fig = px.scatter(
        df_perf_bin,
        # x="precision_0",
        x="accuracy",
        y="accuracy",
        size="size",
        color="model_cat",
        hover_name="model",
        hover_data=["model", "accuracy"],
        log_x=False,
        size_max=25,
        title="Accuracy d'ensemble",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Scatter plot de comparaison des performances
    data = df_perf_bin[col_metrics]
    data = data[data["model"].isin(top_models)].set_index("model")
    fig2 = px.line(
        data.T, markers=True, title="Métriques pour une sélection de modèles"
    )
    fig2.update_yaxes(tick0=0, dtick=0.1)
    fig2.add_hline(y=0.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

    st.write("### Résultats comparés - Classification Multi-Classe")
    # données performance sur multi-classes
    df_perf_mcl = load_data("data_mcl_perf.csv")

    # métrique utilisée
    metrics = ["model", "accuracy", "f1_macro", "recall_macro", "precision_macro"]
    models_list = [
        "log_reg_bsl",
        "lsvc_bsl",
        "Logistic_reg_tfidf",
        "LinearSVC_tfidf",
        "GRU_mcl0",
        "GRU_mcl1",
        "GRU_mcl2",
    ]

    # Lineplot des performance
    data_mcl = df_perf_mcl[metrics]
    data_mcl = data_mcl[data_mcl["model"].isin(models_list)].set_index("model")
    fig = px.line(
        data_mcl.T, markers=True, title="Métriques pour une sélection de modèles"
    )
    fig.update_yaxes(tick0=0, dtick=0.1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # Affichage au choix des matrices de confusion
    st.markdown("Afficher la matrice de confusion d'un modèle de votre choix:")
    # [TO DO TO DO ----------]
    # [TO DO TO DO ----------]

    st.write("### Prédictions")
    # Exemples de commentaires dont on veut prédire le sentiment :
    comment_1 = """Je suis très satisfait de mon achat. La livraison est rapide. 
                   Je recommande vivement la redoute."""
    comment_2 = """Je suis très déçus du produit reçu. Le colis était abimé et la livraison avait beaucoup de retard. 
                    Impossible de joindre le service client."""
    comment_3 = "J'ai reçu mon produit sans plus. rien à signaler"

    # chargement d'un modèle de réseau de neuronnes
    @st.cache_data
    def load_nn_model(model_name):
        model_saved = tf.keras.models.load_model(model_name)
        return model_saved

    # chargement d'un modèle sklearn
    @st.cache_data
    def load_pkl_model(model_file):
        return pickle.load(open(model_file, "rb"))

    model_gru = load_nn_model("gru_0_text_vect")
    model_tfidf = load_pkl_model("tfidf_grid_lreg.pkl")

    # options de choix du commentaire à prédire
    options = ["", comment_1, comment_2, comment_3]
    # menu de sélection du commentaire
    comment_selected = st.selectbox("##### Sélectionnez un commentaire", options)
    if comment_selected == "":
        prediction_gru = ""
        prediction_tfidf = ""
    else:
        tos = np.array([comment_selected])
        # prediction du sentiment pour le commentaire sélectionné
        prediction_gru = 1 * (model_gru.predict([tos]) > 0.5)[:, 0]
        prediction_tfidf = model_tfidf.predict(tos)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Prédiction GRU:")
        st.write(prediction_gru)
    with col2:
        st.markdown("##### Prédiction TFIDF - Regression Log.")
        st.write(prediction_tfidf)

if page == pages[4]:
    st.write("## Analyse de Sentiment")
    st.write(
        "Il est important -  avant de conduire ce type d’analyse - **de préparer le texte** afin qu’il soit le plus exploitable possible aux différents outils disponibles."
    )
    st.write(
        "On parle de pre-processing notamment en utilisant des techniques de **racinisation, tokenisation, utilisation d'expressions regulieres et suppression des stop words**."
    )

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

    df = pd.read_csv("redoute_v3.csv")
    # df = df.head(20)

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

    def apply_regex(chaine):
        tokenizer = RegexpTokenizer("[a-zA-Zé]{4,}")
        tokens_regex = tokenizer.tokenize(chaine)
        return " ".join(tokens_regex)  # pour rejoindre les tokens en une chaîne

    # Appliquation de la fonction à la colonne 'comment_cleaned'
    df["comment_cleaned"] = df["comment_cleaned"].apply(apply_regex)

    # Charger le modèle de sentiment BERT
    # model = TFBertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = TFBertForSequenceClassification.from_pretrained(
        "/Users/dkcentral/Documents/500_fmt_ml_inge/01_DataScientist/99_pjt_fil_rouge/streamlit_fusion/Bert"
    )
    # tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    tokenizer = BertTokenizer.from_pretrained(
        "/Users/dkcentral/Documents/500_fmt_ml_inge/01_DataScientist/99_pjt_fil_rouge/streamlit_fusion/Bert"
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
    selected_comment = df.loc[selected_comment_index, "comment"]
    st.subheader("Commentaire sélectionné :")
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
