# -*- coding: utf-8 -*-
"""
Trains a TF-IDF and Logistic Regression baseline model for emotion classification
using the trimmed 7000-sample dataset.
This script covers data loading, preprocessing, training, and evaluation.
Framework: Scikit-learn
"""

# --- 1. IMPORTS ---
import time
import re
import string
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# --- 2. CONFIGURATION & CONSTANTS ---
DATASET_PATH = 'Emotion_Dataset__Imperfectly_Balanced__7000_Tweets_.csv'
SLANG_DICT_PATH = 'kamus_gabungan_bersih.csv'
VECTORIZER_SAVE_PATH = 'tfidf_vectorizer_7000.joblib'
MODEL_SAVE_PATH = 'logistic_regression_model_7000.joblib'

# Model Hyperparameters
MAX_FEATURES = 10000 # Max vocabulary size for TF-IDF
NGRAM_RANGE = (1, 2) # Use unigrams and bigrams

# --- 3. HELPER FUNCTIONS ---

def load_slang_dictionary():
    """Loads the slang dictionary from the specified path."""
    print("Initializing slang dictionary...")
    try:
        slang_df = pd.read_csv(SLANG_DICT_PATH, sep=';', header=None, names=['slang', 'formal'])
        return dict(zip(slang_df['slang'], slang_df['formal']))
    except FileNotFoundError:
        print(f"ERROR: Slang dictionary '{SLANG_DICT_PATH}' not found.")
        return None

def clean_text(text, slang_dict):
    """
    Cleans text by lowercasing, normalizing slang, removing URLs, punctuation, and numbers.
    """
    if not isinstance(text, str): return ""
    text = text.lower()
    # Normalize slang words
    words = [slang_dict.get(word, word) for word in text.split()]
    text = " ".join(words)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.strip()

# --- 4. MAIN EXECUTION ---

def main():
    """Main function to run the entire training and evaluation pipeline."""
    # Step 1: Load Data and Tools
    print("--- STEP 1: LOADING DATA & TOOLS ---")
    slang_dict = load_slang_dictionary()
    if slang_dict is None: return
        
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"ERROR: Dataset '{DATASET_PATH}' not found.")
        return
    print("Data and tools loaded successfully.\n")

    # Step 2: Preprocess Text
    print("--- STEP 2: PREPROCESSING TEXT DATA ---")
    start_preprocess_time = time.time()
    df['cleaned_text'] = df['tweet'].apply(lambda x: clean_text(x, slang_dict))
    print(f"Preprocessing finished in {time.time() - start_preprocess_time:.2f} seconds.\n")

    # Step 3: Prepare Data and Split
    print("--- STEP 3: PREPARING DATA FOR MODEL ---")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    X = df["cleaned_text"].values
    y = df['label_encoded'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split complete: {len(X_train)} train, {len(X_test)} test samples.\n")

    # Step 4: Feature Extraction with TF-IDF
    print("--- STEP 4: VECTORIZING TEXT WITH TF-IDF ---")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Text vectorization complete.\n")
    
    # Step 5: Train Logistic Regression Model
    print("--- STEP 5: TRAINING LOGISTIC REGRESSION MODEL ---")
    training_start_time = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print(f"Training finished in {time.time() - training_start_time:.2f} seconds.\n")

    # Save the model and the vectorizer
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'")
    print(f"Vectorizer saved to '{VECTORIZER_SAVE_PATH}'.\n")

    # Step 6: Evaluation
    print("--- STEP 6: EVALUATING MODEL ON TEST SET ---")
    y_pred = model.predict(X_test_tfidf)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()

    print("\nScript finished successfully!")

if __name__ == '__main__':
    main()
