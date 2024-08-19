import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./Soil_temperature_1.csv')  # Чтение csv файла
print(df)  # Покажет таблицу
print(df.shape)  # Покажет размеры сразу по двум осям
data_january = df[['depth, m', 'Jan t']]  # Смотрим данные только за январь
print(data_january)
print(df.dtypes) # Смотрим типы данных
print(df.idxmin())
print(df.loc[3]) # Получив доступ к строке можем посмотреть температуры  в разные месяцы на определенной глубине

temp_soil = {'depth_m': [0, -0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, -4.5, -5, -5.5, -6],
             'temperature_1': [1.0, 0.5, 0.2, 0, 1, 0, -0, 3, -0.8, -1.3, -1, 9, -2.3],
             'temperature_2': [0.6, 0.1, -0.2, -0.8, -0.2, 0, 0.4, 0.9, 1, 4, 2.1, 2.0, 1.8]}
s = pd.DataFrame(data=temp_soil)
print(s)
print(s.columns)
print(s.mean())

data = np.array([[1, 2], [5, 3], [4, 6]])
print(data.max(axis=0))
print(data.max(axis=1))
m = np.array([[1, 2], [3, 4]])
j = np.array([[1, 1], [1, 1]])
print(m+j)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X = np.arange(-5, 5, 0.2)
Y = np.arange(-5, 5, 0.2)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
plt.savefig('1.png')

