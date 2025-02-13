# Analyse et Prédiction des Sentiments Clients

## 📌 Objectif du Projet

Ce projet vise à :
- Prédire les notations des clients.
- Catégoriser les commentaires.
- Proposer des réponses automatiques.
- Analyser les sentiments des avis clients.

## 🚀 Approches Adoptées

### 1️⃣ Modèles de Classification
Utilisation de divers modèles pour prédire le sentiment client en se basant sur :
- La longueur du commentaire.
- Le titre.
- La durée depuis la transaction.

**Résultat :** Une régression logistique simple a montré des performances notables.

### 2️⃣ Bag of Words avec TF-IDF et Count Vectorizer
L’application du Bag of Words combiné avec TF-IDF et Count Vectorizer a permis une **bonne précision globale**.  
Cependant, **des limitations subsistent** dans la prédiction des sentiments négatifs.

### 3️⃣ Réseaux de Neurones Récurrents (GRU)
Les **GRU** ont surpassé les modèles précédents, atteignant une **précision globale de plus de 91%**,  
améliorant notamment la prédiction des **sentiments négatifs**.

📌 **Note :** Le rééchantillonnage et la prise en compte du poids des classes ont été essentiels pour équilibrer les données.

## 🔍 Analyse des Retours Fournisseurs
Une approche parallèle a été adoptée, incluant :
- **Feature engineering.**
- **Recherche d'une trame standard.**
- **Fine-tuning des modèles.**

**Observation :** L’analyse a révélé des tendances spécifiques telles qu’une **standardisation des réponses**.

## 🧠 Exploration des Sentiments
Différentes techniques ont été testées :
- **Word clouds** pour une première visualisation.
- **BERT** pour capturer les nuances des sentiments.
- **Modèles supervisés et semi-supervisés** comme Naïve Bayes et Topic Modeling.

## 🤖 Intégration d’un Chatbot Interactif
Un chatbot a été mis en place pour :
- **Évaluer les sentiments des commentaires.**
- **Présenter les quatre principaux sujets sous-jacents.**
- **Faciliter l’analyse des avis clients.**

## 📊 Résultats et Perspectives
✔️ Un **système efficace** de prédiction des évaluations clients.  
✔️ Une **interface interactive** pour exploiter les résultats.  
✔️ **Améliorations possibles** : affiner les modèles et intégrer des analyses plus ciblées  
  👉 Diriger automatiquement les commentaires vers les **départements concernés**.

---

📌 **Conclusion :** Ce projet met en évidence l'importance de l'IA dans l’analyse des sentiments clients et ouvre la voie à des améliorations continues pour optimiser la prise de décision.

🚀 **Améliorations futures** : Affinement des paramètres et modélisation plus précise des sujets.

---

## 📁 Structure du Projet



Project Organization
------------

    --- 0_data_collection :
    |
    |        - 01_mdp_trustedshops_scraping_v1.0.ipynb
    |   
    
    --- 1_eda_dataviz
    |
    |        - 01_mdp_eda_redoute_reviews_1.0.ipynb
    |        - 02_mib_eda_redoute_suppliers_1.0.ipynb
    |        - 02_mib_eda_statistics_test_redoute_suppliers_2.0.ipynb 
    |        - 03_mdp_eda_redoute_reviews_nlp_1.0.ipynb
    |  
    
    --- 2_models
    |      
    |--- Modèle baseline : 
    |         - 01_baselines_models_04.ipynb
    |         - 01_baselines_multiclass_02.ipynb
    |    
    |--- Modèles Bag of Words
    |        - 02_text_processing_0.0.ipynb
    |        - 02_vectorization_clf_multiclass_0.1.ipynb
    |        - 02_vectorization_cvz_clf_binary_0.4.ipynb
    |        - 02_vectorization_nn_models_02.ipynb
    |        - 02_vectorization_tfidf_clf_binary_0.4.ipynb
    |        - 02_vectorization_tfidf_gridscv_clf_binary_0.4.ipynb
    |    
    |--- Réseaux de neurones récurrents GRU
    |        - 03_Sentiment_analysis_with_RNN_04.ipynb
    |        - 03_Text_Processing_for_RNN.ipynb
    |    
    |--- Analyse de sentiments TexBlob, Vader, Bert
    |        - 04_Sentiment_Analysis_tb_vd_0.2.ipynb
    |        - 04_Sentiment_analysis_BERT_0.0.ipynb
    |    
    |--- Topic Modeling
    |        - 05_topic_modeling_de_BERTopic.ipynb
    |        - 05_topic_modeling_gensim.0.0.ipynb
    |   
    |--- DataViz
    |         - 06_dviz_models_performance_02.ipynb
    |
   
