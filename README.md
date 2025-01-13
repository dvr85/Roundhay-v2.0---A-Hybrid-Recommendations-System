# Roundhay-v2.0--A-Hybrid-Recommendations-System
Roundhay is a weighted hybrid recommendations system using Ensemble Learning. This repository provides an overview of how the system works and important concepts used throughout the project.

---

## Overview

Recommender systems can generally be classified into three types:

1. **Content-Based Filtering**  
   Recommends items that share attributes or features with items the user has liked in the past.

2. **Collaborative Filtering**  
   Recommends items based on user preferences:
   - **User-based**: Finds users with similar likes and suggests items they also liked.
   - **Item-based**: Finds items that are frequently liked together.

3. **Hybrid Systems**  
   Combines the strengths of content-based and collaborative filtering to produce better recommendations.

### Pros and Cons of Hybrid Systems

- **Pros**  
  - **Accuracy**: Often yield better recommendations than standalone systems.  
  - **Robustness**: Combining multiple methods helps minimize the weaknesses of each individual technique.

- **Cons**  
  - **Complexity**: Can be computationally expensive to implement and maintain multiple systems.  
  - **Tuning**: Requires careful optimization of thresholds or weights to achieve the best performance.



In the previous version of Roundhay, I have used simple weighted hybrid recommendation system which calculates the total weighted score of content based filtering and collaborative filtering using cosine similarity and one-hot encoding method for text processing for content-based filtering where the categorical data like movie genres would be transformed into a binary outcomes (0/1), representing a unique genre.


In this version, I have implemented an another version of weighted hybrid recommendation system which uses the ensemble learning methods (Gradient Boosting) for score prediction and GloVe embeddings for advanced text processing.

---

## Ensemble learning with Gradient Boosting and learned embeddings for Hybrid Recommendation Systems

Ensemble methods combine multiple models to improve the final output which potentially reduces overfitting and variance. Gradient boosting is a powerful ensemble technique that builds models sequentially, each new model correcting errors made by the previously trained models.

Here, instead of cosine similarity, the gradient boosting (XGBoost) integrates the collabrative filtering and content based filtering scores into a single predictive model to estimate final score for a new user.



Extracting meaningful features from text using natural language prcessing techniques like GloVe word embeddings. 

For example, in this recommendation system, I have used movie descriptions available in IMDb links to enhance the content-based part of the system by importing using GloVe embeddings manually. This conversion allows the system to quantify the similarity between movies based on their content descriptions.

---

## Glossary

### A) XGBoost (eXtreme Gradient Boosting)

XGBoost is an ensemble learning method that builds multiple models (called trees) sequentially, where each new model attempts to correct the errors of the previous model, for speed and performance. The models are combined to produce a final prediction of the weighted score.

* XGBoost can provide highly accurate recommendations that are informed by both user behavior and content characteristics.

* It improces the robustness of the recommendation system by making it more effective at handling complexities in real-world datasets.

### B) GloVe Embeddings

GloVe embeddings or Global Vectors for Word Representation is an unsupervised learning algorithm for obtaining vector representations for words, developed by Stanford University.
This model efficiently aggregate the global word-to-word co-occurance to predict the presence of words in the context of others, which helps it capture both the syntac and semantics of words.

* By providing rich, pre-trained word vectors, GloVe enables the system to understand and utilize the semantic and contextual nuances within movie descriptions. This capability significantly improves the accuracy and relevance of movie recommendations based on content similarity, helping to address challenges like the cold start problem for new or less-rated movies.

---

## Dataset

The dataset used in this project is from **MovieLens**, which provides a large set of user ratings and metadata for movies: [MovieLens Dataset](https://grouplens.org/datasets/movielens/)

GloVe embeddings for text processing: [GloVe embeddings] (https://nlp.stanford.edu/projects/glove/)

---



