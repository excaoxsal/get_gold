from sklearn.linear_model import LinearRegression
import numpy as np

# Anggap X adalah urutan hari (1, 2, 3...) dan y adalah harga
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1400000, 1410000, 1405000, 1420000, 1430000])

model = LinearRegression()
model.fit(X, y)

# Prediksi harga untuk hari ke-6
prediksi_besok = model.predict([[6]])
print(f"Prediksi harga besok: Rp{int(prediksi_besok[0]):,}")