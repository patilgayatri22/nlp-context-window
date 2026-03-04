# IMDB Sentiment Analysis using BERT Embeddings + SVM

## Overview

This project performs sentiment classification on the IMDB movie reviews dataset using transformer-based sentence embeddings and traditional machine learning classifiers.

Instead of training deep learning models from scratch, we leverage pre-trained BERT sentence embeddings and train classical ML models (SVM, Logistic Regression) on top of them for efficient and high-performing sentiment classification.

---

## 🧠 Architecture



---

## Features

- Transformer-based sentence embeddings (`sentence-transformers`)
- Support Vector Machine (SVM) classifier
- Logistic Regression baseline
- Balanced dataset sampling
- Modular ML pipeline
- Reproducible experiments
- Clean evaluation metrics (Accuracy, F1, MCC)
- EDA with word clouds and frequency plots

---

## Tech Stack

- Python 3.9+
- sentence-transformers
- scikit-learn
- pandas
- numpy
- nltk
- matplotlib
- seaborn
- TextBlob
- wordcloud
- joblib

---

## Project Structure

imdb/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── IMDB_DataAnalysis.ipynb
│ └── imdb_code.ipynb
│
├── models/
│ └── saved_model.joblib
│
├── imdb_readme.md
├── requirements.txt

---

## Running the Project

Run the notebooks in order:
1) IMDB_DataAnalysis.ipynb → Data Exploration
2) imdb_code.ipynb → Model Training & Evaluation



---

## Exploratory Data Analysis

- Text cleaning and normalization
- Word frequency analysis
- Top word visualization
- Word cloud generation
- Sentiment polarity inspection

---

## Model Training

1. Load and preprocess IMDB dataset  
2. Remove single-sentence documents (if applicable)  
3. Generate sentence embeddings using Sentence-BERT  
4. Aggregate sentence vectors into document vectors  
5. Train classifier (SVM / Logistic Regression)  
6. Evaluate performance  

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Installation

```bash
git clone repo
cd imdb
pip install -r requirements.txt 
```

## Future Improvements

Hyperparameter tuning (GridSearchCV)
Cross-validation
Model versioning
MLflow experiment tracking
FastAPI deployment for inference
Docker containerization
CI/CD integration