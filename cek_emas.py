import requests
from bs4 import BeautifulSoup

def get_harga_emas_antam():
    url = "https://www.logammulia.com/id/harga-emas-hari-ini"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.find_all('tr')
        
        data_emas = []
        # Mengambil dari baris ke-2 (index 1) untuk melewati header tabel
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) >= 2:
                teks_gram = cols[0].text.strip()
                # Bersihkan angka dari 'Rp', '.', dan spasi agar bisa diolah
                teks_harga = cols[1].text.strip().replace('Rp', '').replace('.', '').replace(',', '').strip()
                
                if teks_harga.isdigit():
                    data_emas.append({
                        "gram": teks_gram,
                        "harga": int(teks_harga)
                    })
        
        return data_emas # Mengembalikan LIST of dictionaries

    except Exception as e:
        return {"error": str(e)}

# --- JALANKAN DAN TAMPILKAN ---
hasil = get_harga_emas_antam()

# 1. Cek jika terjadi error koneksi/request
if isinstance(hasil, dict) and "error" in hasil:
    print(f"Terjadi kesalahan: {hasil['error']}")

# 2. Cek jika data kosong (struktur web mungkin berubah)
elif not hasil:
    print("Gagal mengambil data: Tabel tidak ditemukan atau kosong.")

# 3. Jika berhasil, tampilkan semua baris menggunakan looping
else:
    print(f"{'UKURAN':<15} | {'HARGA (Rp)':<15}")
    print("-" * 35)
    for item in hasil:
        # Menggunakan format :, untuk pemisah ribuan agar rapi
        print(f"{item['gram']:<15} | {item['harga']:>12,}")

    # Contoh jika ingin mengambil spesifik baris ke-2 saja (indeks 1)
    if len(hasil) > 1:
        print(f"\nCatatan: Harga emas ukuran {hasil[1]['gram']} adalah Rp{hasil[1]['harga']:,}")