import sqlite3
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# --- 1. Fungsi Inisialisasi Database ---
def init_db():
    conn = sqlite3.connect('monitoring_emas.db')
    cursor = conn.cursor()
    # Membuat tabel jika belum ada
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS harga_harian (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT,
            gram TEXT,
            harga INTEGER
        )
    ''')
    conn.commit()
    return conn

# --- 2. Fungsi Scraper (Versi Anda yang sudah diperbaiki) ---
def get_harga_emas_antam():
    url = "https://www.logammulia.com/id/harga-emas-hari-ini"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')
        
        data_emas = []
        
        for row in rows:
            cols = row.find_all('td')
            
            # Pastikan ini adalah baris data (memiliki kolom td)
            if len(cols) >= 2:
                teks_gram = cols[0].text.strip().lower() # Ubah ke lowercase untuk pencocokan aman
                
                # Filter hanya untuk "1 gr"
                if teks_gram == "1 gr":
                    # Ambil Harga Dasar (kolom index 1)
                    raw_harga = cols[1].text.strip()
                    
                    # Bersihkan karakter non-angka (Rp, titik, koma)
                    # Menggunakan join & filter agar lebih robust terhadap format ribuan
                    teks_harga = ''.join(filter(str.isdigit, raw_harga))
                    
                    if teks_harga:
                        # Kita hanya ambil satu data lalu break (berhenti) 
                        # agar tidak mengambil "1 gr" dari Gift Series di bawahnya
                        data_emas.append(("1 gr", int(teks_harga)))
                        break 
                        
        return data_emas
    except Exception as e:
        print(f"Error scraping: {e}")
        return []

# --- 3. Eksekusi Utama ---
if __name__ == "__main__":
    # Ambil data terbaru
    daftar_harga = get_harga_emas_antam()
    
    if daftar_harga:
        conn = init_db()
        cursor = conn.cursor()
        tgl_sekarang = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Simpan semua pecahan emas ke database
        for gram, harga in daftar_harga:
            cursor.execute('''
                INSERT INTO harga_harian (tanggal, gram, harga, sumber)
                VALUES (?, ?, ?, ?)
            ''', (tgl_sekarang, gram, harga, "logammulia.com"))
        
        conn.commit()
        conn.close()
        print(f"Sukses menyimpan {len(daftar_harga)} data pada {tgl_sekarang}")