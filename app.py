import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Jabodetabek House Price Estimator",
    page_icon="🏡",
    layout="wide"
)

# -----------------------------------------------------------------
# --- LOAD MODEL & DATA ARTIFACTS ---
# -----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model_harga_rumah.pkl')
        columns = joblib.load('model_columns.pkl')
        return model, columns
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('jabodetabek_house_price.csv')
        # Preprocessing dasar untuk visualisasi
        feature = ['city', 'certificate', 'land_size_m2', 'building_size_m2', 'bedrooms', 'bathrooms', 'garages', 'carports', 'price_in_rp']
        df = df[feature].dropna(subset=['price_in_rp'])
        # Filter batasan logika rumah (hapus anomali ekstrem)
        df = df[(df['bedrooms'] <= 10) & (df['bathrooms'] <= 10) & (df['garages'] <= 5) & (df['carports'] <= 5)]
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None

model, model_columns = load_artifacts()
df_history = load_dataset()

if model is None or df_history is None:
    st.stop()

# -----------------------------------------------------------------
# UI - SIDEBAR (Input Parameter)
# -----------------------------------------------------------------
st.sidebar.header("🏡 Spesifikasi Properti")

# Biarkan UI tetap rapi untuk user
city = st.sidebar.selectbox(
    "Kota (Wilayah):",
    ["Jakarta Selatan", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara", "Jakarta Pusat", "Tangerang", "Bekasi", "Depok", "Bogor"]
)

# Gunakan teks yang lebih profesional di UI
certificate = st.sidebar.selectbox(
    "Tipe Sertifikat:",
    ["SHM - Sertifikat Hak Milik", "HGB - Hak Guna Bangunan", "Lainnya"]
)

# --- Mapping Sertifikat ke format asli dataset ---
cert_mapping = {
    "SHM - Sertifikat Hak Milik": "shm - sertifikat hak milik",
    "HGB - Hak Guna Bangunan": "hgb - hak guna bangunan",
    "Lainnya": "lainnya (ppjb,girik,adat,dll)"
}

st.sidebar.markdown("---")
st.sidebar.subheader("Dimensi Bangunan")
# Tambahkan step=10 agar saat diklik panah atas/bawah, langsung loncat 10 meter persegi
land_size = st.sidebar.number_input("Luas Tanah (m²)", min_value=10, max_value=5000, value=120, step=10)
building_size = st.sidebar.number_input("Luas Bangunan (m²)", min_value=10, max_value=5000, value=90, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Fasilitas Ruangan")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    bedrooms = st.number_input("Kamar Tidur", min_value=1, max_value=10, value=3)
    garages = st.number_input("Garasi (Mobil)", min_value=0, max_value=5, value=1)
with col_sb2:
    bathrooms = st.number_input("Kamar Mandi", min_value=1, max_value=10, value=2)
    carports = st.number_input("Carport (Mobil)", min_value=0, max_value=5, value=0)

# -----------------------------------------------------------------
# --- PROSES PREDIKSI UTAMA ---
# -----------------------------------------------------------------
# Siapkan data input
# -----------------------------------------------------------------
# --- PROSES PREDIKSI UTAMA ---
# -----------------------------------------------------------------
# Siapkan data input
input_data = pd.DataFrame({
    'land_size_m2': [np.log1p(land_size)], 
    'building_size_m2': [np.log1p(building_size)], 
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'garages': [garages],
    'carports': [carports],
    'city': [f" {city}"], # <--- Tambahkan spasi di depan agar cocok dengan dataset
    'certificate': [cert_mapping[certificate]] # <--- Gunakan nama asli dari dataset
})

# Encoding & Alignment (Tambahkan dtype=int agar tidak error true/false)
input_encoded = pd.get_dummies(input_data, dtype=int)
input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

# Encoding & Alignment
input_encoded = pd.get_dummies(input_data)
input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

# Prediksi
prediksi_log = model.predict(input_final)
prediksi_rupiah = np.expm1(prediksi_log[0])

# Kalkulasi Statistik Kota
avg_city_price = df_history[df_history['city'] == city]['price_in_rp'].mean()
if pd.isna(avg_city_price):
    avg_city_price = df_history['price_in_rp'].mean() # Fallback

# -----------------------------------------------------------------
# UI - HALAMAN UTAMA
# -----------------------------------------------------------------
st.title("🏡 Smart Forecast: Harga Rumah Jabodetabek")
st.subheader(f"Analisis Estimasi Properti di {city}")
st.markdown("---")

# -----------------------------------------------------------------
# --- OUTPUT METRIK ---
# -----------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Prediksi Nilai Pasar Wajar",
        value=f"Rp {prediksi_rupiah/1e9:.2f} M",
        delta="Berdasarkan algoritma Random Forest"
    )

with col2:
    selisih_avg = prediksi_rupiah - avg_city_price
    st.metric(
        label=f"Rata-rata Harga di {city}",
        value=f"Rp {avg_city_price/1e9:.2f} M",
        delta=f"{selisih_avg/1e9:+.2f} M vs rata-rata",
        delta_color="inverse"
    )

with col3:
    # Logika Penilaian Status Harga
    st.markdown("**Status Valuasi Properti:**")
    if prediksi_rupiah <= avg_city_price * 0.8:
        st.success("✅ **Sangat Terjangkau (Di bawah pasar)**")
    elif prediksi_rupiah <= avg_city_price * 1.5:
        st.info("⚖️ **Harga Kompetitif (Sesuai pasar)**")
    else:
        st.warning("👑 **Properti Premium (Di atas rata-rata)**")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------
# --- VISUALISASI GAUGE CHART ---
# -----------------------------------------------------------------
st.subheader("Visualisasi Posisi Harga")

max_gauge = max(avg_city_price * 2.5, prediksi_rupiah * 1.5)

fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = prediksi_rupiah,
    title = {'text': "Estimasi Harga (Rupiah)"},
    delta = {'reference': avg_city_price, 'position': "top"},
    gauge = {
        'axis': {'range': [0, max_gauge]},
        'bar': {'color': "#1E3A8A"},
        'steps' : [
            {'range': [0, avg_city_price * 0.8], 'color': "rgba(46, 139, 87, 0.3)"}, # Hijau (Murah)
            {'range': [avg_city_price * 0.8, avg_city_price * 1.5], 'color': "rgba(255, 165, 0, 0.3)"}, # Oranye (Wajar)
            {'range': [avg_city_price * 1.5, max_gauge], 'color': "rgba(255, 0, 0, 0.3)"} # Merah (Premium)
        ],
        'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prediksi_rupiah}
    }
))
fig_gauge.update_layout(height=400)
st.plotly_chart(fig_gauge, use_container_width=True)

st.info("""
**Cara Membaca Grafik Posisi Harga:**
* **Jarum Hitam / Angka Utama:** Estimasi harga spesifik untuk properti yang Anda inputkan.
* **Area Hijau:** Zona properti dengan harga di bawah rata-rata kota (relatif terjangkau).
* **Area Oranye:** Zona harga kompetitif standar di wilayah tersebut.
* **Area Merah:** Zona properti premium / mewah.
* **Angka Delta (Segitiga):** Perbandingan harga rumah Anda dengan rata-rata keseluruhan properti di kota tersebut.
""")

# -----------------------------------------------------------------
# --- TABEL DATA HISTORIS ---
# -----------------------------------------------------------------
with st.expander("Lihat Dataset Historis Listing Jabodetabek"):
    st.dataframe(df_history.head(1000), use_container_width=True)

# -----------------------------------------------------------------
# --- EVALUASI MODEL (SCATTER PLOT AKTUAL VS PREDIKSI) ---
# -----------------------------------------------------------------
st.markdown("---")
st.header("🔬 Evaluasi Performa Machine Learning")

with st.spinner("Menghitung ulang metrik evaluasi..."):
    # Kalkulasi Ulang Cepat untuk Visualisasi (Menggunakan dataset yang sudah di-drop NA)
    X_viz = df_history.drop('price_in_rp', axis=1)
    y_viz = np.log1p(df_history['price_in_rp'])
    
    X_viz['land_size_m2'] = np.log1p(X_viz['land_size_m2'])
    X_viz['building_size_m2'] = np.log1p(X_viz['building_size_m2'])
    
    X_viz_encoded = pd.get_dummies(X_viz)
    X_viz_final = X_viz_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Split kecil untuk mendapatkan test set
    _, X_test_eval, _, y_test_eval = train_test_split(X_viz_final, y_viz, test_size=0.2, random_state=42)
    
    y_pred_eval_log = model.predict(X_test_eval)
    
    y_test_real = np.expm1(y_test_eval)
    y_pred_real = np.expm1(y_pred_eval_log)
    
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)

col_eval1, col_eval2, col_eval3 = st.columns(3)
col_eval1.metric("MAE (Rata-rata Error)", f"Rp {mae/1e6:.1f} Juta")
col_eval2.metric("RMSE", f"Rp {rmse/1e6:.1f} Juta")
col_eval3.metric("R² Score (Akurasi)", f"{r2:.4f}")

st.subheader("Grafik Prediksi vs Harga Aktual (Data Test)")

fig_eval = go.Figure()

fig_eval.add_trace(go.Scatter(
    x=y_test_real,
    y=y_pred_real,
    mode='markers',
    name='Data Properti',
    marker=dict(color='#10B981', size=8, opacity=0.6)
))

min_val = min(y_test_real.min(), y_pred_real.min())
max_val = max(y_test_real.max(), y_pred_real.max())

fig_eval.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Garis Sempurna',
    line=dict(color='red', dash='dash')
))

fig_eval.update_layout(
    xaxis_title="Harga Properti Aktual (Rp)",
    yaxis_title="Harga Prediksi Model (Rp)",
    title=f"Korelasi Aktual vs Prediksi (R² = {r2:.2f})",
    showlegend=True,
    height=500
)

st.plotly_chart(fig_eval, use_container_width=True)

st.info("""
**Cara Membaca Grafik Evaluasi:**
* **Sumbu X:** Harga asli rumah di pasar (Listing).
* **Sumbu Y:** Harga yang ditebak oleh algoritma Random Forest.
* **Garis Merah Putus-putus:** Target ideal (Jika model menebak 100% benar, semua titik akan berada di garis ini).
* Semakin dekat titik-titik hijau menyelimuti garis merah, berarti model semakin akurat memahami pola harga rumah.
""")