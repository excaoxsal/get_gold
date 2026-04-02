import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

print("📥 1. Mengambil data dari SQLite...")
conn = sqlite3.connect('monitoring_emas.db')
df = pd.read_sql_query("SELECT tanggal, harga FROM harga_harian WHERE harga IS NOT NULL", conn)
conn.close()

# --- PREPROCESSING ---
print("⚙️ 2. Memproses dan membersihkan data...")
df['tanggal'] = pd.to_datetime(df['tanggal'])
df = df.groupby('tanggal')['harga'].mean().reset_index() # Atasi duplikat
df = df.sort_values('tanggal').set_index('tanggal')
df = df.resample('D').ffill() # Isi tanggal bolong/hari libur

# Ambil nilai harganya saja
dataset = df['harga'].values
dataset = dataset.reshape(-1, 1)

print("⚖️ 3. Menormalisasi Data (Skala 0 sampai 1)...")
# LSTM sangat sensitif terhadap angka besar, kita ubah jutaan rupiah jadi angka 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# --- MEMBUAT DATASET UNTUK LSTM ---
# Kita gunakan 60 hari ke belakang (lookback) untuk memprediksi 1 hari ke depan
lookback = 60 
x_train, y_train = [], []

for i in range(lookback, len(scaled_data)):
    x_train.append(scaled_data[i-lookback:i, 0]) # Memori masa lalu (xt, ht-1)
    y_train.append(scaled_data[i, 0])            # Target jawaban (Harga aktual)

x_train, y_train = np.array(x_train), np.array(y_train)

# Format ulang x_train menjadi 3 Dimensi [jumlah_sampel, langkah_waktu, fitur]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# --- MEMBANGUN MODEL (ARSITEKTUR OTAK) ---
print("🧠 4. Membangun Arsitektur LSTM...")
model = Sequential()

# Layer LSTM Pertama (Return sequences=True agar memori dilanjutkan ke layer berikutnya)
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Mematikan 20% neuron acak agar mesin tidak sekadar menghafal (Overfitting)

# Layer LSTM Kedua
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Layer Output (Dense Layer) untuk menghasilkan 1 angka tebakan harga
model.add(Dense(units=1))

# Compile model (Menentukan cara mesin menghitung error dan memperbaiki bobot/Weight)
model.compile(optimizer='adam', loss='mean_squared_error')

# --- PROSES BELAJAR (TRAINING) ---
print("🏋️ 5. Memulai Proses Belajar (Training)... Mohon tunggu sebentar.")
# Epochs = Berapa kali mesin membaca seluruh data
# Batch_size = Mesin mengevaluasi per 32 baris data sekaligus
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

# --- MEMPREDIKSI MASA DEPAN ---
print("🔮 6. Memprediksi harga untuk 30 hari ke depan...")
hari_kedepan = 30
# Ambil 60 hari terakhir dari data historis untuk pijakan awal tebakan
memori_terakhir = scaled_data[-lookback:] 
batch_prediksi = memori_terakhir.reshape(1, lookback, 1)

prediksi_masa_depan = []

# Loop untuk menebak hari demi hari ke depan
for i in range(hari_kedepan):
    # Tebak harga besok
    tebakan_besok = model.predict(batch_prediksi, verbose=0)[0]
    prediksi_masa_depan.append(tebakan_besok)
    
    # Masukkan tebakan besok ke dalam memori, dan buang memori hari paling tua (hari ke-1)
    # Ini agar mesin bisa menebak lusa menggunakan data hari ini + tebakan besok
    batch_prediksi = np.append(batch_prediksi[:, 1:, :], [[tebakan_besok]], axis=1)

# Denormalisasi (Kembalikan angka 0-1 menjadi skala Rupiah)
prediksi_rupiah = scaler.inverse_transform(prediksi_masa_depan)

# --- VISUALISASI ---
print("📊 7. Menyiapkan Grafik...")
# Buat deret tanggal untuk masa depan
tanggal_terakhir = df.index[-1]
tanggal_prediksi = pd.date_range(start=tanggal_terakhir + pd.Timedelta(days=1), periods=hari_kedepan, freq='D')

# Ambil data historis setahun terakhir saja agar grafik tidak terlalu padat
df_historis = df.last('365D')

plt.figure(figsize=(14, 7))
plt.plot(df_historis.index, df_historis['harga'], label='Data Aktual Historis', color='blue')
plt.plot(tanggal_prediksi, prediksi_rupiah, label=f'Prediksi LSTM {hari_kedepan} Hari', color='red', linestyle='dashed', linewidth=2)

plt.title('Prediksi Harga Emas Antam dengan Machine Learning (LSTM)', fontsize=14)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Harga (Rp)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Format axis Y agar tidak jadi angka eksponensial (e.g., 1e6)
plt.ticklabel_format(style='plain', axis='y')
current_yticks = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_yticks])

plt.tight_layout()
plt.show()

print("\n✨ Proses selesai! Silakan lihat grafik prediksi Anda.")