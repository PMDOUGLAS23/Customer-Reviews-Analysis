Supply Chain - Satisfaction des clients
==============================
**Objectifs du projet****
- Prédire la satisfaction client à partir des commentaires. 
- De ces commentaires, identifier les catégories de sujets problématiques 
A partir des commentaires clients,  être capable d’automatiser une réponse
Détection du sentiment client : positif, neutre ou négatif - à confirmer


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
   
