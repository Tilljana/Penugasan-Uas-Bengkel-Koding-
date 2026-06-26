import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ================================================================
# CONFIG
# ================================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ================================================================
# LOAD ARTIFACTS
# ================================================================
# Sesuaikan path ini dengan struktur folder lokal / GitHub Anda
MODEL_PATH = 'models/'

@st.cache_resource
def load_artifacts():
    try:
        model            = joblib.load(os.path.join(MODEL_PATH, 'rf_preprocessing_model.pkl'))
        scaler           = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        features         = joblib.load(os.path.join(MODEL_PATH, 'rf_prep_features.pkl'))
        label_encoders   = joblib.load(os.path.join(MODEL_PATH, 'label_encoders.pkl'))
        return model, scaler, features, label_encoders
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.info("Pastikan 4 file .pkl sudah di-download dari Colab dan dimasukkan ke folder 'models/'")
        st.stop()

model, scaler, features, label_encoders = load_artifacts()

# ================================================================
# UI HEADER
# ================================================================
st.title("📊 Prediksi Customer Churn (Random Forest)")
st.markdown("Aplikasi web ini menggunakan model **Random Forest (Tahap Preprocessing)**.")
st.markdown("---")

# ================================================================
# DYNAMIC INPUT FORM
# ================================================================
st.subheader("📋 Input Data Pelanggan")
st.markdown("Isi form di bawah ini untuk memprediksi probabilitas churn.")

input_data = {}

# Bagi form ke dalam 3 kolom agar rapi
cols = st.columns(3)

# Mengisi form secara dinamis berdasarkan urutan fitur yang diharapkan model
for i, feat in enumerate(features):
    col = cols[i % 3]
    with col:
        # Jika fitur tersebut adalah fitur kategorikal (ada di label_encoders)
        if feat in label_encoders:
            # Ambil semua kategori unik dari encoder untuk dijadikan dropdown
            options = label_encoders[feat].classes_
            # Ambil mode (nilai default/terbanyak) sebagai default dropdown
            input_data[feat] = st.selectbox(
                label=feat.replace('_', ' ').title(),
                options=options,
                key=feat
            )
        else:
            # Jika fitur numerik, gunakan input angka
            # Tipe default float agar bisa menampung desimal
            input_data[feat] = st.number_input(
                label=feat.replace('_', ' ').title(),
                value=0.0,
                format="%.2f",
                key=feat
            )

st.markdown("---")

# ================================================================
# PREDIKSI
# ================================================================
if st.button("🔍 Cek Status Pelanggan", type="primary", use_container_width=True):
    
    # 1. Masukkan input dari user ke DataFrame satu baris
    input_df = pd.DataFrame([input_data])
    
    # 2. Lakukan Encoding untuk fitur kategorikal
    for col in input_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Tangani kemungkinan teks tidak dikenal dengan fallback ke kelas pertama
            input_val = input_df[col].iloc[0]
            if input_val in le.classes_:
                encoded_val = le.transform([input_val])[0]
            else:
                encoded_val = 0 
            input_df[col] = encoded_val
            
    # Pastikan tipe datanya numerik agar tidak error saat di-scale
    input_df = input_df.astype(float)
    
    # 3. Lakukan Scaling menggunakan scaler dari Colab
    # Penting: Pastikan urutan kolom sesuai dengan saat training
    input_scaled = scaler.transform(input_df[features])
    
    # 4. Lakukan Prediksi
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    churn_prob = proba[1] * 100
    
    # 5. Tampilkan Hasil
    st.subheader("📈 Hasil Analisis")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        if prediction == 1:
            st.error("### ⚠️ CHURN")
            st.markdown("Pelanggan **berpotensi tinggi** meninggalkan layanan.")
        else:
            st.success("### ✅ AMAN (TIDAK CHURN)")
            st.markdown("Pelanggan **cenderung bertahan**.")
            
    with res_col2:
        st.markdown("#### Probabilitas Churn")
        st.progress(int(churn_prob))
        
        if churn_prob >= 70:
            st.error(f"🔴 Risiko Tinggi: **{churn_prob:.1f}%**")
        elif churn_prob >= 40:
            st.warning(f"🟡 Risiko Sedang: **{churn_prob:.1f}%**")
        else:
            st.success(f"🟢 Risiko Rendah: **{churn_prob:.1f}%**")
