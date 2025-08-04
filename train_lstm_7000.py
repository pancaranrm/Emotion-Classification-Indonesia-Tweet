# -*- coding: utf-8 -*-
"""
Trains a Bidirectional LSTM model for emotion classification on the trimmed 7000-sample dataset.
This script uses PyTorch and includes text-based percentage progress updates.
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
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from collections import Counter

# --- 2. CONFIGURATION & CONSTANTS ---
DATASET_PATH = 'Emotion_Dataset__Imperfectly_Balanced__7000_Tweets_.csv'
SLANG_DICT_PATH = 'kamus_gabungan_bersih.csv'
MODEL_SAVE_PATH = 'lstm_emotion_model_7000.pth'

# Model Hyperparameters
VOCAB_SIZE = 10000
MAX_LENGTH = 128
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
OUTPUT_DIM = 5 # Number of emotion classes
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3


# --- 3. HELPER FUNCTIONS ---

def load_and_prepare_tools():
    """Initializes and returns the Sastrawi stemmer, stopword set, and slang dictionary."""
    print("Initializing NLP tools (Stemmer, Stopwords, Slang Dictionary)...")
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_set = set(stopword_factory.get_stop_words()) - {"tidak", "tapi", "begitu"}
    try:
        slang_df = pd.read_csv(SLANG_DICT_PATH, sep=';', header=None, names=['slang', 'formal'])
        slang_dict = dict(zip(slang_df['slang'], slang_df['formal']))
    except FileNotFoundError:
        print(f"ERROR: Slang dictionary '{SLANG_DICT_PATH}' not found.")
        return None, None, None
    return stemmer, stopword_set, slang_dict

def clean_text(text, stemmer, stopword_set, slang_dict):
    """Cleans and preprocesses a single text entry, returning a list of tokens."""
    if not isinstance(text, str): return []
    text = text.lower()
    words = [slang_dict.get(word, word) for word in text.split()]
    text = " ".join(words)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [stemmer.stem(word) for word in text.split() if word not in stopword_set]
    return tokens

def build_vocab(tokenized_texts, max_vocab_size):
    """Builds a vocabulary from tokenized texts."""
    word_counts = Counter(word for text in tokenized_texts for word in text)
    most_common_words = [word for word, count in word_counts.most_common(max_vocab_size - 2)]
    word_to_idx = {word: i + 2 for i, word in enumerate(most_common_words)}
    word_to_idx['<PAD>'] = 0
    word_to_idx['<OOV>'] = 1
    return word_to_idx

def texts_to_sequences(tokenized_texts, word_to_idx):
    """Converts texts to integer sequences using the vocabulary."""
    return [[word_to_idx.get(word, word_to_idx['<OOV>']) for word in text] for text in tokenized_texts]

def pad_sequences(sequences, max_len):
    """Pads sequences to a fixed length."""
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if seq_len > max_len:
            padded[i, :] = seq[:max_len]
        else:
            padded[i, :seq_len] = seq
    return padded

def show_progress(iteration, total, prefix=''):
    """Displays a simple text-based percentage that updates on the same line."""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    sys.stdout.write(f'\r{prefix}... {percent}% Complete   ')
    sys.stdout.flush()

# --- 4. PYTORCH LSTM MODEL DEFINITION ---

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # *2 for bidirectional

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


# --- 5. MAIN EXECUTION ---

def main():
    """Main function to run the entire training and evaluation pipeline."""
    print("--- STEP 1: LOADING DATA & TOOLS ---")
    stemmer, stopword_set, slang_dict = load_and_prepare_tools()
    if not all([stemmer, stopword_set, slang_dict]): return
    
    df = pd.read_csv(DATASET_PATH)
    print("Data and tools loaded successfully.\n")

    print("--- STEP 2: PREPROCESSING TEXT DATA ---")
    start_preprocess_time = time.time()
    cleaned_tokens = []
    total_rows = len(df)
    for i, text in enumerate(df['tweet']):
        cleaned_tokens.append(clean_text(text, stemmer, stopword_set, slang_dict))
        if (i + 1) % 10 == 0 or (i + 1) == total_rows:
             show_progress(i + 1, total_rows, prefix='Cleaning data')
    df['cleaned_tokens'] = cleaned_tokens
    print(f"\nPreprocessing finished in {time.time() - start_preprocess_time:.2f} seconds.\n")

    print("--- STEP 3: PREPARING DATA FOR MODEL ---")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['label'])
    
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(df['cleaned_tokens'], labels, test_size=0.2, random_state=42, stratify=labels)
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(X_train_texts, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    word_to_idx = build_vocab(X_train_texts, VOCAB_SIZE)
    X_train_seq = texts_to_sequences(X_train_texts, word_to_idx)
    X_val_seq = texts_to_sequences(X_val_texts, word_to_idx)
    X_test_seq = texts_to_sequences(X_test_texts, word_to_idx)

    X_train_pad = pad_sequences(X_train_seq, MAX_LENGTH)
    X_val_pad = pad_sequences(X_val_seq, MAX_LENGTH)
    X_test_pad = pad_sequences(X_test_seq, MAX_LENGTH)
    
    print(f"Data split complete: {len(X_train_pad)} train, {len(X_val_pad)} validation, {len(X_test_pad)} test samples.\n")

    # Step 4: Create PyTorch DataLoaders
    print("--- STEP 4: CREATING PYTORCH DATALOADERS ---")
    train_data = TensorDataset(torch.from_numpy(X_train_pad), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_val_pad), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test_pad), torch.from_numpy(y_test))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    print("DataLoaders created.\n")

    # Step 5: Model Training
    print("--- STEP 5: MODEL TRAINING ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Ensure OUTPUT_DIM matches the number of unique classes
    num_classes = len(label_encoder.classes_)
    model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    training_start_time = time.time()
    
    for epoch in range(EPOCHS):
        print(f'======== Epoch {epoch + 1} / {EPOCHS} ========')
        model.train()
        total_loss, total_acc = 0, 0
        num_train_batches = len(train_loader)
        for i, (texts, labels) in enumerate(train_loader):
            show_progress(i + 1, num_train_batches, prefix='Training')
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            total_acc += torch.sum(preds == labels).item()
            
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(y_train)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        
        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        num_val_batches = len(val_loader)
        with torch.no_grad():
            for i, (texts, labels) in enumerate(val_loader):
                show_progress(i + 1, num_val_batches, prefix='Validation')
                texts, labels = texts.to(device), labels.to(device)
                output = model(texts)
                loss = criterion(output, labels)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                val_acc += torch.sum(preds == labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(y_val)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        print(f"\n  Training Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    print(f"\nTotal training time: {time.time() - training_start_time:.2f} seconds")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'.\n")

    # Step 6: Final Evaluation
    print("--- STEP 6: FINAL EVALUATION ON TEST SET ---")
    model.eval()
    predictions, true_labels = [], []
    num_test_batches = len(test_loader)
    for i, (texts, labels) in enumerate(test_loader):
        show_progress(i + 1, num_test_batches, prefix='Evaluating')
        texts = texts.to(device)
        output = model(texts)
        preds = torch.argmax(output, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.numpy())

    print("\n\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))
    
    conf_matrix = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); plt.show()
    
    print("\n--- STEP 7: VISUALIZING TRAINING HISTORY ---")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nScript finished successfully!")

if __name__ == '__main__':
    main()
