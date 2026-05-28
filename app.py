import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

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
        eval_data = joblib.load('eval_data.pkl')
        return model, columns, eval_data
    except Exception as e:
        st.error(f"Gagal memuat file artifact model (Pastikan model_harga_rumah.pkl, model_columns.pkl, dan eval_data.pkl ada di folder): {e}")
        return None, None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('jabodetabek_house_price.csv')
        feature = ['city', 'certificate', 'land_size_m2', 'building_size_m2', 'bedrooms', 'bathrooms', 'garages', 'carports', 'price_in_rp']
        df = df[feature].dropna(subset=['price_in_rp'])
        
        def remove_outliers_iqr_local(data, columns):
            df_out = data.copy()
            for col in columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
            return df_out

        df = remove_outliers_iqr_local(df, ['price_in_rp', 'land_size_m2', 'building_size_m2', 'bedrooms', 'bathrooms', 'garages', 'carports'])
        return df
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        return None

model, model_columns, eval_data = load_artifacts()
df_history = load_dataset()

if model is None or df_history is None or eval_data is None:
    st.stop()

# -----------------------------------------------------------------
# UI - SIDEBAR (Input Parameter)
# -----------------------------------------------------------------
st.sidebar.header("🏡 Spesifikasi Properti")

city = st.sidebar.selectbox(
    "Kota (Wilayah):",
    ["Jakarta Selatan", "Jakarta Barat", "Jakarta Timur", "Jakarta Utara", "Jakarta Pusat", "Tangerang", "Bekasi", "Depok", "Bogor"]
)

certificate = st.sidebar.selectbox(
    "Tipe Sertifikat:",
    ["SHM - Sertifikat Hak Milik", "HGB - Hak Guna Bangunan", "Lainnya"]
)

cert_mapping = {
    "SHM - Sertifikat Hak Milik": "shm - sertifikat hak milik",
    "HGB - Hak Guna Bangunan": "hgb - hak guna bangunan",
    "Lainnya": "lainnya (ppjb,girik,adat,dll)"
}

st.sidebar.markdown("---")
st.sidebar.subheader("Dimensi Bangunan")
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

# Fungsi pembaca format mata uang Indonesia
def format_rupiah(angka):
    if angka >= 1e9:
        return f"Rp {angka/1e9:.2f} Miliar"
    elif angka >= 1e6:
        return f"Rp {angka/1e6:.0f} Juta"
    else:
        return f"Rp {angka:,.0f}"

# -----------------------------------------------------------------
# UI - HALAMAN UTAMA
# -----------------------------------------------------------------
st.title("🏡 Smart Forecast: Harga Rumah Jabodetabek")
st.subheader(f"Analisis Estimasi Properti di {city}")
st.markdown("---")

# -----------------------------------------------------------------
# --- PROSES PREDIKSI & OUTPUT METRIK ---
# -----------------------------------------------------------------
if st.button("Hitung Estimasi Harga Rumah", type="primary", use_container_width=True):
    
    # 🛑 BLOK VALIDASI INPUT USER
    if building_size > (land_size * 3):
        st.error("❌ Luas bangunan terlalu tidak masuk akal (lebih dari 3x lipat luas tanah). Silakan perbaiki input Anda.")
        st.stop()
    elif building_size > land_size:
        st.warning("⚠️ Catatan: Luas bangunan Anda lebih besar dari luas tanah. Kami mengasumsikan ini adalah rumah bertingkat (2 lantai atau lebih).")
        
    with st.spinner("Model sedang menganalisis spesifikasi properti..."):
        # Siapkan data input
        input_data = pd.DataFrame({
            'land_size_m2': [np.log1p(land_size)], 
            'building_size_m2': [np.log1p(building_size)], 
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'garages': [garages],
            'carports': [carports],
            'city': [f" {city}"], 
            'certificate': [cert_mapping[certificate]]
        })

        input_encoded = pd.get_dummies(input_data, dtype=int)
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

        # Jalankan Prediksi
        prediksi_log = model.predict(input_final)
        prediksi_rupiah = np.expm1(prediksi_log[0])

        # Kalkulasi Statistik Kota
        avg_city_price = df_history[df_history['city'] == f" {city}"]['price_in_rp'].mean()
        if pd.isna(avg_city_price):
            avg_city_price = df_history['price_in_rp'].mean()

        # Menampilkan Output Metrik
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Prediksi Nilai Pasar Wajar",
                value=format_rupiah(prediksi_rupiah),
                delta="Algoritma Random Forest"
            )

        with col2:
            selisih_avg = prediksi_rupiah - avg_city_price
            st.metric(
                label=f"Rata-rata Harga Historis di {city}",
                value=format_rupiah(avg_city_price),
                delta=f"{selisih_avg/1e9:+.2f} M vs rata-rata",
                delta_color="inverse"
            )

        with col3:
            st.markdown("**Status Valuasi Properti:**")
            if prediksi_rupiah <= avg_city_price * 0.8:
                st.success("✅ **Ekonomis (Di bawah pasar)**")
            elif prediksi_rupiah <= avg_city_price * 1.5:
                st.info("⚖️ **Kompetitif (Wajar/Sesuai pasar)**")
            else:
                st.warning("👑 **Premium (Di atas rata-rata)**")

        st.markdown("<br>", unsafe_allow_html=True)

        # Menampilkan Visualisasi Gauge Chart
        st.subheader("Visualisasi Posisi Harga")
        max_gauge = max(avg_city_price * 2.5, prediksi_rupiah * 1.5)

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prediksi_rupiah,
            title = {'text': "Taksiran Nilai Jual (Rp)"},
            delta = {'reference': avg_city_price, 'position': "top"},
            gauge = {
                'axis': {'range': [0, max_gauge]},
                'bar': {'color': "#1E3A8A"},
                'steps' : [
                    {'range': [0, avg_city_price * 0.8], 'color': "rgba(46, 139, 87, 0.3)"},
                    {'range': [avg_city_price * 0.8, avg_city_price * 1.5], 'color': "rgba(255, 165, 0, 0.3)"},
                    {'range': [avg_city_price * 1.5, max_gauge], 'color': "rgba(255, 0, 0, 0.3)"}
                ],
                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prediksi_rupiah}
            }
        ))
        fig_gauge.update_layout(height=380)
        st.plotly_chart(fig_gauge, use_container_width=True)

# -----------------------------------------------------------------
# --- TABEL DATA & EVALUASI (TETAP MUNCUL DI LUAR TOMBOL) ---
# -----------------------------------------------------------------
st.markdown("---")

with st.expander("Lihat Sampel Dataset Historis (Data Pasca IQR)"):
    st.dataframe(df_history.head(500), use_container_width=True)

st.header("🔬 Evaluasi Performa Kualitas Model")

# Ekstrak data hasil evaluasi original milik notebook
y_test_real = eval_data['y_aktual']
y_pred_real = eval_data['y_prediksi']
mae = eval_data['mae']
rmse = eval_data['rmse']
r2 = eval_data['r2']

col_eval1, col_eval2, col_eval3 = st.columns(3)
col_eval1.metric("MAE (Mean Absolute Error)", f"Rp {mae:,.2f}")
col_eval2.metric("RMSE (Root Mean Squared Error)", f"Rp {rmse:,.2f}")
col_eval3.metric("R² Score (Akurasi Global)", f"{r2:.4f}")

st.subheader("Grafik Prediksi vs Harga Aktual (Data Test Sesuai Notebook)")

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
    title=f"Korelasi Aktual vs Prediksi Real-Time (R² = {r2:.4f})",
    showlegend=True,
    height=500
)

st.plotly_chart(fig_eval, use_container_width=True)