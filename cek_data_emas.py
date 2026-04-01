import pandas as pd
import sqlite3

conn = sqlite3.connect('monitoring_emas.db')
df = pd.read_sql_query("SELECT * FROM harga_harian WHERE gram = '1 gr'", conn)
print(df.head()) # Data siap dianalisis!