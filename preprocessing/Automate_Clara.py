import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

def automate_clara(path):
    print("=== MEMUAT DATASET ===")
    df = pd.read_csv(path)
    print(df.head())

    print("\n=== INFORMASI DATA ===")
    print(df.info())
    print("Shape:", df.shape)
    print("Columns:", df.columns)

    # Set kolom teks & label
    text_col = 'title'
    label_col = 'real'
    print(f"\nSet text_col={text_col} dan label_col={label_col}")

    # Distribusi Label
    print("\n=== DISTRIBUSI LABEL ===")
    print(df[label_col].value_counts(dropna=False))

    sns.countplot(x=df[label_col])
    plt.title("Distribusi Label")
    plt.show()

    # Missing values
    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum())

    # Duplicate rows
    print("\n=== DUPLIKAT ===")
    print("Jumlah Duplikat:", df.duplicated().sum())

    # Drop kolom tidak dibutuhkan
    print("\nMenghapus kolom 'news_url' ...")
    df = df.drop(columns=['news_url'])

    # Mengisi missing source_domain
    print("Mengisi missing value pada 'source_domain' ...")
    df['source_domain'] = df['source_domain'].fillna("Unknown")
    print(df.isnull().sum())

    # Menghapus duplikat
    print("\nMenghapus baris duplikat ...")
    df = df.drop_duplicates()
    print("Duplikat setelah dibersihkan:", df.duplicated().sum())

    # Preprocessing Teks
    print("\n=== PREPROCESSING TEKS ===")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

        return " ".join(tokens)

    print("Membersihkan teks pada kolom:", text_col)
    df['clean_title'] = df[text_col].astype(str).apply(clean_text)

    # Tampilkan contoh cleaning
    print("\n=== CONTOH SEBELUM & SESUDAH CLEANING ===")
    print("------------------------------------------")

    for i in range(3):
        print(f"\nOriginal: {df[text_col].iloc[i]}")
        print(f"Cleaned : {df['clean_title'].iloc[i]}")

    # Simpan dataset hasil preprocessing
    save_path = "FakeNewsNet_cleaned.csv"
    df.to_csv(save_path, index=False)
    print(f"\nDataset berhasil disimpan sebagai: {save_path}")

    return df


# Jalankan otomatis ketika file dipanggil
if __name__ == "__main__":
    path = "FakeNewsNet_raw.csv"  # sesuaikan sesuai lokasi file
    automate_clara(path)