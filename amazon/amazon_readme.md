# Amazon Reviews Sentiment Analysis using BERT Embeddings + Machine Learning

## Overview

This project performs sentiment analysis on Amazon product reviews using transformer-based sentence embeddings combined with classical machine learning models.

Instead of training deep neural networks from scratch, we leverage pre-trained Sentence-BERT embeddings and train efficient ML classifiers such as:

- Support Vector Machine (SVM)
- Logistic Regression

The pipeline includes:
- Text cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Sentence embedding generation
- Document-level vector aggregation
- Model training and evaluation

---

## Key Features

- Transformer-based embeddings (`sentence-transformers`)
- Balanced dataset handling
- Modular ML pipeline
- Classical ML classifiers on top of BERT
- Detailed evaluation metrics
- Word frequency and word cloud visualization
- Reproducible results

---

## Tech Stack

- Python 3.9+
- sentence-transformers
- PyTorch
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

amazon/
│
├── data/
│
├── scripts/
│ ├── AmazonReview_EDA.ipynb
│ └── amazon_reviews.ipynb
│
├── models/
│ └── saved_model.joblib
│
├── amazon_readme.md
├── requirements.txt


---

## Exploratory Data Analysis (EDA)

The project includes:

- Text cleaning and normalization
- Stopword removal
- Word frequency distribution
- Top words visualization
- Word cloud generation
- Sentiment polarity inspection
- Class imbalance analysis

---

## Model Training Pipeline

1. Load Amazon review dataset
2. Clean and preprocess review text
3. Remove unnecessary or short entries
4. Generate sentence embeddings using Sentence-BERT
5. Aggregate sentence embeddings to document-level vectors
6. Train classifier (SVM / Logistic Regression)
7. Evaluate performance on test data

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
git clone https://github.com/yourusername/amazon-review-sentiment.git
cd amazon-review-sentiment
pip install -r requirements.txt
```