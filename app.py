import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Prediksi Emas AI", layout="centered")

st.title("📈 Prediksi Harga Emas AI")
st.markdown("Aplikasi web ini menggunakan **Long Short-Term Memory (LSTM)** untuk memprediksi harga emas Antam selama 6 bulan ke depan.")

# 2. FUNGSI AMBIL DATA
@st.cache_data
def load_data():
    conn = sqlite3.connect('monitoring_emas.db')
    df = pd.read_sql_query("SELECT tanggal, harga FROM harga_harian WHERE harga IS NOT NULL", conn)
    conn.close()
    
    df['tanggal'] = pd.to_datetime(df['tanggal'], format='mixed')
    df = df.groupby('tanggal')['harga'].mean().reset_index()
    df = df.sort_values('tanggal').set_index('tanggal')
    df = df.resample('D').ffill()
    return df

# 3. FUNGSI TRAINING SEKALIGUS PREDIKSI (Di-cache agar kebal error Threading)
@st.cache_data(show_spinner=False)
def proses_ai_dan_prediksi(_df, hari_kedepan=180):
    dataset = _df['harga'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    lookback = 60
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Proses Training AI
    model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Langsung jalankan Prediksi 6 Bulan di sini
    memori_terakhir = scaled_data[-lookback:]
    batch_prediksi = memori_terakhir.reshape(1, lookback, 1)

    prediksi_masa_depan = []
    for i in range(hari_kedepan):
        tebakan = model.predict(batch_prediksi, verbose=0)[0]
        prediksi_masa_depan.append(tebakan)
        batch_prediksi = np.append(batch_prediksi[:, 1:, :], [[tebakan]], axis=1)

    # Kembalikan ke angka Rupiah
    prediksi_rupiah = scaler.inverse_transform(prediksi_masa_depan)
    return prediksi_rupiah


# --- TAMPILAN USER INTERFACE (UI) ---
df = load_data()
st.info(f"Berhasil memuat **{len(df)}** baris data historis dari database.")

# Tombol Eksekusi
if st.button("Jalankan Prediksi 6 Bulan", type="primary"):
    hari_kedepan = 180
    
    # Spinner akan muncul saat pertama kali diklik (AI belajar),
    # Untuk klik kedua, ketiga dst, akan langsung melompat instan karena mengambil dari Cache
    with st.spinner('🤖 AI sedang menganalisis pola historis dan meracik hasil ramalan... (Mohon tunggu ±1-2 menit)'):
        prediksi_rupiah = proses_ai_dan_prediksi(df, hari_kedepan)

    # Rendering Grafik
    tanggal_terakhir = df.index[-1]
    tanggal_prediksi = pd.date_range(start=tanggal_terakhir + pd.Timedelta(days=1), periods=hari_kedepan, freq='D')
    
    # Memperbaiki error Pandas (menggunakan tail sebagai ganti last)
    df_historis = df.tail(365) 

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_historis.index, df_historis['harga'], label='Data Aktual (1 Tahun)', color='blue')
    ax.plot(tanggal_prediksi, prediksi_rupiah, label='Prediksi 6 Bulan', color='red', linestyle='dashed')

    ax.set_title('Proyeksi AI: Harga Emas Antam')
    ax.set_ylabel('Harga (Rp)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Format angka agar rapi
    ax.ticklabel_format(style='plain', axis='y')
    y_ticks = ax.get_yticks()
    ax.set_yticklabels(['{:,.0f}'.format(x) for x in y_ticks])

    # Tampilkan ke Web
    st.pyplot(fig)
    st.success("✨ Prediksi selesai dimuat! Anda bebas merefresh atau menekan tombol berulang kali.")
    
    # Mencegah memory leak dari matplotlib
    plt.close(fig)