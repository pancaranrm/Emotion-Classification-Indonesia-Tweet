# -*- coding: utf-8 -*-
"""
Fine-tunes an IndoBERT model for emotion classification using the trimmed 7000-sample dataset.
This script covers data loading, preprocessing, training, evaluation, and visualization.
Framework: PyTorch
Model: indobenchmark/indobert-base-p2
"""

# --- 1. IMPORTS ---
import time
import re
import string
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

# --- 2. CONFIGURATION & CONSTANTS ---
# Updated file paths to use the new dataset
DATASET_PATH = 'Emotion_Dataset__Imperfectly_Balanced__7000_Tweets_.csv'
SLANG_DICT_PATH = 'kamus_gabungan_bersih.csv'
MODEL_NAME = "indobenchmark/indobert-base-p2"
MODEL_SAVE_PATH = './fine_tuned_indobert_7000'

# Training Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
EPSILON = 1e-8


# --- 3. HELPER FUNCTIONS ---

def load_and_prepare_tools():
    """Initializes and returns the Sastrawi stemmer, stopword set, and slang dictionary."""
    print("Initializing NLP tools (Stemmer, Stopwords, Slang Dictionary)...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_set = set(stopword_factory.get_stop_words()) - {"tidak", "tapi", "begitu"}
    try:
        # Assuming the new slang dictionary has the same format
        slang_df = pd.read_csv(SLANG_DICT_PATH, sep=';', header=None, names=['slang', 'formal'])
        slang_dict = dict(zip(slang_df['slang'], slang_df['formal']))
    except FileNotFoundError:
        print(f"ERROR: Slang dictionary '{SLANG_DICT_PATH}' not found.")
        return None, None, None
    return stemmer, stopword_set, slang_dict

def clean_text(text, stemmer, stopword_set, slang_dict):
    """Cleans and preprocesses a single text entry."""
    if not isinstance(text, str): return ""
    text = text.lower()
    words = [slang_dict.get(word, word) for word in text.split()]
    text = " ".join(words)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [stemmer.stem(word) for word in text.split() if word not in stopword_set]
    return ' '.join(tokens)

def create_pytorch_dataloader(encodings, labels, sampler_class):
    """Creates a PyTorch DataLoader from tokenized encodings and labels."""
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    sampler = sampler_class(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

def show_progress(iteration, total, prefix=''):
    """Displays a simple text-based percentage that updates on the same line."""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    sys.stdout.write(f'\r{prefix}... {percent}% Complete   ')
    sys.stdout.flush()


# --- 4. MAIN EXECUTION ---

def main():
    """Main function to run the entire training and evaluation pipeline."""
    # Step 1: Load Data and Tools
    print("--- STEP 1: LOADING DATA & TOOLS ---")
    stemmer, stopword_set, slang_dict = load_and_prepare_tools()
    if not all([stemmer, stopword_set, slang_dict]): return
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"ERROR: Dataset '{DATASET_PATH}' not found.")
        return
    print("Data and tools loaded successfully.\n")

    # Step 2: Preprocess Text
    print("--- STEP 2: PREPROCESSING TEXT DATA ---")
    start_preprocess_time = time.time()
    cleaned_texts = []
    total_rows = len(df)
    for i, text in enumerate(df['tweet']):
        cleaned_texts.append(clean_text(text, stemmer, stopword_set, slang_dict))
        if (i + 1) % 10 == 0 or (i + 1) == total_rows:
             show_progress(i + 1, total_rows, prefix='Cleaning data')
    df['cleaned_text'] = cleaned_texts
    print(f"\nPreprocessing finished in {time.time() - start_preprocess_time:.2f} seconds.\n")

    # Step 3: Prepare Data for Modeling
    print("--- STEP 3: PREPARING DATA FOR MODEL ---")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    X = df["cleaned_text"].values
    y = df['label_encoded'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    print(f"Data split complete: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.\n")

    # Step 4: Tokenization
    print("--- STEP 4: TOKENIZING DATA ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(X_train.tolist(), max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
    val_encodings = tokenizer(X_val.tolist(), max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
    test_encodings = tokenizer(X_test.tolist(), max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt')
    print("Tokenization complete.\n")

    # Step 5: Create DataLoaders
    print("--- STEP 5: CREATING PYTORCH DATALOADERS ---")
    train_dataloader = create_pytorch_dataloader(train_encodings, y_train, RandomSampler)
    val_dataloader = create_pytorch_dataloader(val_encodings, y_val, SequentialSampler)
    test_dataloader = create_pytorch_dataloader(test_encodings, y_test, SequentialSampler)
    print("DataLoaders created.\n")

    # Step 6: Model Training
    print("--- STEP 6: MODEL TRAINING ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    training_start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'======== Epoch {epoch + 1} / {EPOCHS} ========')
        model.train()
        total_train_loss, total_train_accuracy = 0, 0
        
        num_train_batches = len(train_dataloader)
        for i, batch in enumerate(train_dataloader):
            show_progress(i + 1, num_train_batches, prefix='Training')
            b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = outputs.logits.detach().cpu().numpy()
            total_train_accuracy += np.mean(np.argmax(logits, axis=1) == b_labels.to('cpu').numpy())
        
        avg_train_loss = total_train_loss / num_train_batches
        avg_train_acc = total_train_accuracy / num_train_batches
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        print(f"\n  Average training loss: {avg_train_loss:.4f} | Accuracy: {avg_train_acc:.4f}")

        model.eval()
        total_val_loss, total_val_accuracy = 0, 0
        num_val_batches = len(val_dataloader)
        for i, batch in enumerate(val_dataloader):
            show_progress(i + 1, num_val_batches, prefix='Validation')
            b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            total_val_loss += outputs.loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            total_val_accuracy += np.mean(np.argmax(logits, axis=1) == b_labels.to('cpu').numpy())

        avg_val_loss = total_val_loss / num_val_batches
        avg_val_acc = total_val_accuracy / num_val_batches
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        print(f"\n  Validation loss: {avg_val_loss:.4f} | Accuracy: {avg_val_acc:.4f}")

    print(f"\nTotal training time: {time.time() - training_start_time:.2f} seconds")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'.\n")

    # Step 7: Final Evaluation
    print("--- STEP 7: FINAL EVALUATION ON TEST SET ---")
    model.eval()
    predictions, true_labels = [], []
    num_test_batches = len(test_dataloader)
    for i, batch in enumerate(test_dataloader):
        show_progress(i + 1, num_test_batches, prefix='Evaluating')
        b_input_ids, b_attention_mask, b_labels = [t.to(device) for t in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask)
        logits = outputs.logits.detach().cpu().numpy()
        predictions.extend(np.argmax(logits, axis=1))
        true_labels.extend(b_labels.to('cpu').numpy())

    print("\n\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))

    conf_matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.show()

    # Step 8: Visualize Training
    print("\n--- STEP 8: VISUALIZING TRAINING HISTORY ---")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'o-', label='Training Loss')
    plt.plot(history['val_loss'], 'o-', label='Validation Loss')
    plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'o-', label='Training Accuracy')
    plt.plot(history['val_acc'], 'o-', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    
    plt.tight_layout()
    plt.show()

    print("\nScript finished successfully!")


if __name__ == '__main__':
    main()
