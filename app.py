import streamlit as st
import fitz  # PyMuPDF
import joblib
import io
import numpy as np
from function import preprocess_text, highlight_text_in_pdf, kalimat_ke_vektor, get_formal_suggestion
from nltk.tokenize import sent_tokenize
import os

# Load model dan vectorizer
model = joblib.load("./model/svc_w2vec_model_bobot_finall_banget_3.joblib")
processed = ""
# vectorizer = np.array([kalimat_ke_vektor(words) for words in processed])
# vectorizer = joblib.load("vectorizer_svc.joblib")
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.title("Deteksi Gaya Teks PDF (Formal / Non-Formal)")

uploaded_file = st.file_uploader("Upload file PDF", type="pdf")



if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += page.get_text()

    sentences = sent_tokenize(all_text)

    kalimat_non_formal = []
    kalimat_dan_saran = []

    for kalimat in sentences:
        hasil_prep = preprocess_text(kalimat)
        if not hasil_prep:
            continue
        processed = [' '.join(hasil_prep)]
        transformed = np.array([kalimat_ke_vektor(words) for words in processed])
        pred = model.predict(transformed)[0]
        if pred == 1:
            kalimat_non_formal.append(kalimat)
            saran_formal = get_formal_suggestion(kalimat)
            kalimat_dan_saran.append((kalimat, saran_formal))

    hasil_doc = highlight_text_in_pdf(pdf_bytes, kalimat_non_formal)

    pdf_output = io.BytesIO()
    hasil_doc.save(pdf_output)
    pdf_output.seek(0)

    st.download_button(
        label="Download PDF dengan Highlight Non-Formal",
        data=pdf_output.read(),
        file_name="hasil_highlighted.pdf",
        mime="application/pdf"
    )

    st.write(f"Jumlah kalimat non-formal terdeteksi: {len(kalimat_non_formal)}")

    st.write("### Saran Perbaikan Kalimat Non-Formal:")
    for idx, (ori, saran) in enumerate(kalimat_dan_saran, 1):
        st.markdown(f"**{idx}.** `{ori}`\n\n➡️ **Saran Formal:** _{saran}_\n")

