import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit UI untuk mengupload file CSV atau Excel
st.title('Klasifikasi Penyakit PCOS Menggunakan Algoritma Random Forest')

# Upload file CSV atau Excel
uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Membaca file CSV dengan pemisah titik koma
        df = pd.read_csv(uploaded_file, sep=';')

        # Menampilkan data yang diupload
        st.write("Data yang diupload:")
        st.write(df.head())

        # Cek apakah kolom target 'PCOS (Y/N)' ada
        if 'PCOS (Y/N)' not in df.columns:
            st.error("File tidak mengandung kolom 'PCOS (Y/N)'. Harap pastikan nama kolom target sesuai.")
            st.stop()

        # Memisahkan fitur dan target
        X = df.drop(['PCOS (Y/N)', 'Patient File No.'], axis=1, errors='ignore')  # Menghapus kolom yang tidak diperlukan
        y = df['PCOS (Y/N)']

        # Split data menjadi data pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Membuat dan melatih model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluasi model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Menampilkan akurasi model
        st.write(f"Akurasi model: {accuracy * 100:.2f}%")

        # Input data pengguna
        st.header('Masukkan Data Pasien untuk Prediksi')

        # Input sesuai kolom yang ada di dataset
        age = st.text_input("Age (yrs)")
        weight = st.text_input("Weight (Kg)")
        height = st.text_input("Height (Cm)")
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        blood_group = st.selectbox("Blood Group", ["A", "B", "AB", "O"])
        pulse_rate = st.text_input("Pulse rate (bpm)")
        rr = st.text_input("RR (breaths/min)")
        hb = st.number_input("Hb (g/dl)", min_value=5.0, max_value=20.0, value=12.0, step=0.1)
        cycle = st.selectbox("Cycle (R/I)", ["Regular", "Irregular"])
        cycle_length = st.slider("Cycle length (days)", 20, 35, 28)
        marriage_status = st.slider("Marriage Status (Yrs)", 0, 50, 5)
        pregnant = st.selectbox("Pregnant (Y/N)", ["No", "Yes"])
        no_of_abortions = st.slider("No. of abortions", 0, 10, 0)
        hip = st.slider("Hip (inch)", 30, 50, 36)
        waist = st.slider("Waist (inch)", 20, 50, 28)
        waist_hip_ratio = st.number_input("Waist:Hip Ratio", min_value=0.5, max_value=1.5, value=0.75)

        # Buat DataFrame untuk prediksi dengan nama kolom yang sesuai
        input_data = {
            'Age (yrs)': [age],
            'Weight (Kg)': [weight],
            'Height (Cm)': [height],
            'BMI': [bmi],
            'Blood Group': [blood_group],
            'Pulse rate(bpm)': [pulse_rate],
            'RR (breaths/min)': [rr],
            'Hb(g/dl)': [hb],
            'Cycle(R/I)': [cycle],
            'Cycle length(days)': [cycle_length],
            'Marraige Status (Yrs)': [marriage_status],
            'Pregnant(Y/N)': [pregnant],
            'No. of abortions': [no_of_abortions],
            'Hip(inch)': [hip],
            'Waist(inch)': [waist],
            'Waist:Hip Ratio': [waist_hip_ratio]
        }

        # Pastikan input_df memiliki kolom yang sama dengan dataset pelatihan
        input_df = pd.DataFrame(input_data)
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        # Prediksi hasil
        prediction = model.predict(input_df)

        # Tampilkan hasil prediksi
        if prediction[0] == 1:
            st.write("**Diagnosis**: Anda berisiko terkena PCOS.")
        else:
            st.write("**Diagnosis**: Anda tidak terdeteksi PCOS.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
