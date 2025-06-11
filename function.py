import streamlit as st
from nltk.tokenize import sent_tokenize
import re
import fitz  # PyMuPDF
import pandas as pd
from nltk.tokenize import sent_tokenize
from langdetect import detect
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec
from openai import OpenAI
import os
model_w2vec = Word2Vec.load("./model/word2vec_model_final_banget_2.model")




# Hasil kalimat semua jurnal
all_sentences = []



# def hapus_daftar_pustaka(teks):
#     keywords = ['daftar pustaka', 'referensi', 'bibliography', 'references']
#     for keyword in keywords:
#         index = teks.lower().find(keyword)
#         if index != -1:
#             return teks[:index]
#     return teks


# def is_indonesian(sentence):
#     try:
#         return detect(sentence) == 'id'
#     except:
#         return False

# def preprocess_text(text, min_kata=5):
#     text = hapus_daftar_pustaka(text)
#     text = re.sub(r'([a-zA-Z])(\d)(?!\.)', r'\1 \2', text)  # Pisahkan huruf dan angka
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s.,:;()\-/]', '', text)  # Jangan hapus angka dan simbol penting
#     text = re.sub(r'\s+', ' ', text).strip()
#     sentences = sent_tokenize(text)

#     hasil = []
#     for s in sentences:
#         if re.match(r'^(gambar|tabel|bab)\s+\w+[\.:]$', s.strip()):
#             hasil.append(s)
#             continue
#         if len(s.split()) < min_kata:
#             continue
#         if re.fullmatch(r'[\d\s.,:;()\-]+', s.strip()):
#             continue
#         if not is_indonesian(s):
#             continue
#         hasil.append(s)
#     return hasil








def hapus_daftar_pustaka(teks):
    keywords = ['daftar pustaka', 'referensi', 'bibliography', 'references']
    for keyword in keywords:
        index = teks.lower().find(keyword)
        if index != -1:
            return teks[:index]
    return teks


def is_indonesian(sentence):
    try:
        return detect(sentence) == 'id'
    except:
        return False

def preprocess_text(text, min_kata=5):
    text = hapus_daftar_pustaka(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s.,:;()\-/]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = sent_tokenize(text)

    hasil = []
    for s in sentences:
        if len(s.split()) < min_kata:
            continue
        if re.fullmatch(r'[\d\s.,:;()\-]+', s.strip()):
            continue
        if not is_indonesian(s):
            continue
        hasil.append(s)
    return hasil




def highlight_text_in_pdf(pdf_bytes, kalimat_list):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for kalimat in kalimat_list:
        for page in doc:
            areas = page.search_for(kalimat, quads=False)
            for area in areas:
                highlight = page.add_highlight_annot(area)
                highlight.update()
    return doc

def kalimat_ke_vektor(kalimat):
    words = word_tokenize(kalimat.lower())
    vectors = []
    for w in words:
        if w in model_w2vec.wv:
            vectors.append(model_w2vec.wv[w])
    if len(vectors) == 0:
        return np.zeros(model_w2vec.vector_size)
    return np.mean(vectors, axis=0)



# def get_formal_suggestion(nonformal_text):
#     api_key = os.getenv("OPENROUTER_API_KEY")
#     if not api_key:
#         raise ValueError("API key for OpenRouter not set in environment variables.")

#     client = OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=api_key
#     )
#     prompt = f"Tolong ubah kalimat berikut menjadi versi yang lebih formal dalam konteks penulisan artikel jurnal ilmiah. cukup tuliskan alternatif/rekomendasi kalimat dan alasannya, tidak perlu sampai rinci ke kesimpulan dan saran terbaik, cukup berikan saja saran kalimatnya. Jangan memberikan penjelasan yang panjang sehingga membuat sulit membaca rekomendasi kalimatnya. Berikut adalah teksnya:\n\n\"{nonformal_text}\""
    
#     completion = client.chat.completions.create(
#         extra_headers={
#             "HTTP-Referer": "https://yourapp.example",  # opsional
#             "X-Title": "DeteksiKalimat"                 # opsional
#         },
#         model="deepseek/deepseek-r1-0528:free",
#         messages=[
#             {"role": "user", "content": prompt}
#         ]
#     )
    
#     return completion.choices[0].message.content
def get_formal_suggestion(nonformal_text):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Memberikan pesan error jika API key tidak ditemukan
        print("Error: API key untuk OpenRouter tidak diatur dalam environment variables.")
        return "Maaf, API key tidak ditemukan. Mohon atur variabel lingkungan OPENROUTER_API_KEY."

    try:
        # Inisialisasi klien OpenAI dengan base_url OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Prompt yang akan dikirimkan ke model AI
        prompt = f"Tolong ubah kalimat berikut menjadi versi yang lebih formal dalam konteks penulisan artikel jurnal ilmiah. cukup tuliskan alternatif/rekomendasi kalimat dan alasannya, tidak perlu sampai rinci ke kesimpulan dan saran terbaik, cukup berikan saja saran kalimatnya. Jangan memberikan penjelasan yang panjang sehingga membuat sulit membaca rekomendasi kalimatnya. Berikut adalah teksnya:\n\n\"{nonformal_text}\""
        
        # Melakukan panggilan ke API OpenRouter dengan model 'qwen/qwen3-14b:free'
        completion = client.chat.completions.create(
            model="qwen/qwen3-14b:free",  # Model AI yang digunakan (Qwen)
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Memeriksa apakah respons API valid sebelum mencoba mengakses kontennya
        if completion is not None and hasattr(completion, 'choices') and len(completion.choices) > 0:
            return completion.choices[0].message.content
        else:
            # Mengembalikan pesan error jika respons API kosong atau tidak sesuai
            print("Error: Respon API dari OpenRouter kosong atau tidak sesuai format.")
            return "Maaf, saya tidak bisa memberikan saran formal saat ini. Respon AI tidak valid."

    except Exception as e:
        # Menangkap error umum
        print(f"Terjadi error tak terduga saat mendapatkan saran formal: {e}")
        return "Maaf, terjadi kesalahan internal saat memproses permintaan Anda."
