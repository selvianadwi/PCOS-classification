import streamlit as st
import pandas as pd
import joblib

# Load model
def load_model(model_path):
    return joblib.load(model_path)

# Model
model_paths = {
    "Model Original": "models/model1.pkl",
    "Model dengan Feature Selection": "models/model2.pkl",
    "Model dengan SMOTE": "models/model3.pkl",
    "Model dengan SMOTE + Feature Selection": "models/model4.pkl",
    "Model dengan Outlier Removal": "models/model5.pkl",
    "Model dengan Outlier Removal + Feature Selection": "models/model6.pkl",
    "Model dengan Outlier + SMOTE": "models/model7.pkl",
    "Model dengan Outlier + SMOTE + Feature Selection": "models/model8.pkl",
}

# Judul
st.title("PCOS Classification App")

uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, delimiter=';')
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Data yang diunggah:")
    st.dataframe(df.head())

# Pilih model yang akan digunakan
selected_model = st.selectbox("Pilih Model", list(model_paths.keys()))

# Load model yang dipilih
model_path = model_paths[selected_model]
model = load_model(model_path)
feature_names = model.feature_names_in_

# Input data manual
st.subheader("Input Data Manual")
manual_input = {}

# List kolom yang menggunakan selectbox
binary_columns = [
    'Pregnant(Y/N)', 'Weight gain(Y/N)', 'hair growth(Y/N)', 
    'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
    'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)'
]

# Loop untuk membuat input sesuai jenis kolom
for feature in feature_names:
    if feature in binary_columns:
        value = st.selectbox(f"Masukkan nilai {feature}", options=["No", "Yes"])
        # Konversi nilai menjadi numerik
        manual_input[feature] = 1 if value == "Yes" else 0
    else:
        # Input numerik untuk kolom lainnya
        manual_input[feature] = st.number_input(f"Masukkan nilai {feature}", value=0.0)

# Fungsi untuk melakukan prediksi
def predict(input_data, model):
    # Pastikan input_data memiliki nama kolom yang sesuai dengan model
    input_data = pd.DataFrame([input_data])
    input_data.columns = feature_names
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    return prediction[0], prediction_proba[0][1]


if st.button("Diagnosa"):
    diagnosis, probability = predict(manual_input, model)
    accuracy = probability if diagnosis == 1 else (1 - probability)

    st.success(f"Hasil Diagnosis: {'Positive' if diagnosis == 1 else 'Negative'}")
    st.write(f"Akurasi Berdasarkan Input: {accuracy * 100:.2f}%")
    # st.write(f"Probabilitas Diagnosis Positif: {probability * 100:.2f}%")
   
