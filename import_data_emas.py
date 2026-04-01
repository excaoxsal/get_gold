import pandas as pd
import sqlite3
import os
import kagglehub

def process_kaggle_gold(dataset_handle, source_label):
    print(f"\n--- Memproses Dataset: {dataset_handle} ---")
    dataset_path = kagglehub.dataset_download(dataset_handle)
    
    files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not files:
        print(f"⚠️ Tidak ada file CSV ditemukan.")
        return pd.DataFrame()

    all_data = []
    
    for file in files:
        file_path = os.path.join(dataset_path, file)
        print(f"📖 Membaca file: {file}")
        
        try:
            # Baca file (atasi karakter hantu dengan utf-8-sig)
            df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
            
            # 1. Bersihkan karakter aneh di nama kolom asli
            df.columns = df.columns.str.replace(r'[^\x00-\x7F]+', '', regex=True).str.strip().str.lower()
            
            # 2. PEMILIHAN KOLOM AMAN (Mencegah Duplikat Kolom)
            # Cari kolom pertama yang cocok untuk tanggal
            tanggal_col = next((c for c in ['date', 'time (ms)', 'timestamp', 'tanggal', 'waktu'] if c in df.columns), None)
            # Cari kolom pertama yang cocok untuk harga
            harga_col = next((c for c in ['gold price', 'price_buy', 'price', 'close', 'harga'] if c in df.columns), None)
            
            if not tanggal_col or not harga_col:
                print(f"   ⏩ Lewati {file}: Kolom utama tidak cocok. (Tersedia: {df.columns.tolist()})")
                continue
                
            # 3. Potong DataFrame HANYA untuk 2 kolom tersebut
            df = df[[tanggal_col, harga_col]].copy()
            df.columns = ['tanggal', 'harga']
            
            # 4. PEMBERSIHAN TANGGAL
            if tanggal_col == 'time (ms)':
                # Format khusus untuk file raw (Unix milidetik)
                df['tanggal'] = pd.to_datetime(df['tanggal'], unit='ms', errors='coerce')
            else:
                # Format normal (errors='coerce' otomatis mengabaikan baris yang bukan tanggal)
                df['tanggal'] = pd.to_datetime(df['tanggal'].astype(str).str.strip(), errors='coerce')
            
            # 5. PEMBERSIHAN HARGA CERDAS
            # Langkah A: Coba konversi normal dulu (Ini mempertahankan 4000+ data Yudifaturohman)
            harga_num = pd.to_numeric(df['harga'], errors='coerce')
            
            # Langkah B: Jika ada yang gagal konversi (misal karena tulisan "Rp 1.000.000")
            mask_gagal = harga_num.isna() & df['harga'].notna()
            if mask_gagal.any():
                # Hapus SEMUA karakter kecuali angka (buang titik, koma, spasi, Rp)
                clean_str = df.loc[mask_gagal, 'harga'].astype(str).str.replace(r'[^\d]', '', regex=True)
                # Timpa kembali ke variabel angka
                harga_num.loc[mask_gagal] = pd.to_numeric(clean_str, errors='coerce')
                
            df['harga'] = harga_num
            
            # 6. Finalisasi
            df = df.dropna(subset=['tanggal', 'harga']) # Buang baris yang rusak
            
            if not df.empty:
                df['tanggal'] = df['tanggal'].dt.strftime('%Y-%m-%d')
                df['gram'] = '1 gr'
                df['sumber'] = f"{source_label} ({file})"
                
                all_data.append(df[['tanggal', 'gram', 'harga', 'sumber']])
                print(f"   ✅ Berhasil memproses {len(df)} baris.")
            else:
                print(f"   ⚠️ File {file} kosong setelah pembersihan.")
                
        except Exception as e:
            print(f"   ❌ Gagal memproses {file}: {e}")

    # Gabung semua DataFrame jadi satu
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

# --- EKSEKUSI UTAMA ---
db_name = 'monitoring_emas.db'
datasets = [
    {"handle": "garethharrison/antam-historical-gold-price", "label": "Kaggle: garethharrison"},
    {"handle": "yudifaturohman/emas-batangan-antam", "label": "Kaggle: yudifaturohman"}
]

conn = sqlite3.connect(db_name)

try:
    for ds in datasets:
        final_df = process_kaggle_gold(ds["handle"], ds["label"])
        
        if not final_df.empty:
            final_df.to_sql('harga_harian', conn, if_exists='append', index=False)
            print(f"🚀 TOTAL DATA MASUK dari {ds['label']}: {len(final_df)} baris.")

    print("\n✨ SEMUA DATA DARI SEMUA FILE TELAH DISINKRONKAN!")

except Exception as e:
    print(f"\n❌ Terjadi kesalahan sistem: {e}")
finally:
    conn.close()