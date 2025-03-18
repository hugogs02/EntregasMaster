import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tabulate import tabulate
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# Cargamos y formateamos la fecha
viajeros = pd.read_excel('viajerosMD.xlsx')
viajeros['Fecha'] = viajeros['Fecha'].str.strip()
viajeros['Fecha'] = pd.to_datetime(viajeros['Fecha'], format='%YM%m')
viajeros['Viajeros'] = viajeros['Viajeros'].apply(pd.to_numeric, errors='coerce')
viajeros.set_index('Fecha', inplace=True)
viajeros.dropna(inplace=True)
viajeros = viajeros.iloc[::-1]  # Invertimos la serie, ya que los datos van de 2024 para atras

# Hacemos un gráfico con la serie
viajeros.plot(y='Viajeros', figsize=(10,5))
plt.title("Viajeros transportados en Media Distancia")
plt.xlabel("Mes")
plt.ylabel("Viajeros (miles)")
plt.show()

# Representamos los valores por año
viajeros['Año'] = viajeros.index.year.astype(str)
plt.figure(figsize=(12, 8))
sns.lineplot(x=viajeros.index.month, y=viajeros.Viajeros, hue = viajeros['Año'], palette='Spectral')
plt.xlabel('Mes')
plt.ylabel('Estacionalidad')
plt.title('Gráfico estacional por año: Viajeros de Media Distancia')
plt.legend(title='Año', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
viajeros.drop('Año', axis=1, inplace=True) # Eliminamos la columna que habíamos añadido

# Hacemos la descomposición estacional aditiva
add_decomp = seasonal_decompose(viajeros, model='additive', period=12)
plt.rc("figure", figsize=(16,12))
fig = add_decomp.plot()

# Representamos la tendencia y la serie ajustada estacionalmente
viajeros_ajustada = viajeros['Viajeros'] - add_decomp.seasonal
plt.figure(figsize=(12, 8))
plt.plot(viajeros, label='Datos', color='gray')
plt.plot(add_decomp.trend, label='Tendencia', color='blue') # Tendencia
plt.plot(viajeros_ajustada, label='Estacionalmente ajustada', color='red') # Serie ajustada
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
plt.title('Viajeros en Media Distancia')
plt.legend(loc='best')
plt.show()

# Hacemos la descomposición estacional multiplicativa
mult_decomp = seasonal_decompose(viajeros, model='multiplicative', period=12)
plt.rc("figure", figsize=(16,12))
fig = mult_decomp.plot()

# Separamos los datos en train y test, guardando el último año como test
train = viajeros[:-12]
test = viajeros[-12:]

plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='yellow')
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
plt.show()

# Aplicamos el suavizado exponencial simple
model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
fcast = model.forecast(12)

plt.figure(figsize=(12, 8))
# Serie original
plt.plot(train, label='Datos train', color='gray')
plt.plot(test, label='Datos test', color='yellow')
plt.plot(model.fittedvalues, label='Suavizado', color='blue')
plt.plot(fcast, label='Forecast', color='red')
plt.xlabel('Año')
plt.ylabel('Viajeros')
plt.title('Suavizado simple')
plt.legend()
plt.show()

print(model.params_formatted)


# Aplicamos el método de Holt
model1 = Holt(train, initialization_method="estimated").fit()
fcast1 = model1.forecast(12)
print(model1.params_formatted)
print(fcast1)

plt.figure(figsize=(12, 8))
plt.plot(train, label='Datos train', color='gray')
plt.plot(test, label='Datos test', color='yellow')
plt.plot(model1.fittedvalues, label='Suavizado', color='blue')
plt.plot(fcast1, label='Forecast', color='red')
plt.xlabel('Año')
plt.ylabel('Viajeros')
plt.title('Suavizado Holt')
plt.legend()
plt.show()


# Aplicamos el método de la tendencia amortiguada
model2 = Holt(train,damped_trend=True, initialization_method="estimated").fit()
fcast2 = model2.forecast(5)
print(model2.params_formatted)

plt.figure(figsize=(12, 8))
plt.plot(viajeros, label='Datos', color='gray')
plt.plot(fcast, label='ses', color='red')
plt.plot(fcast1, label='Holt', color='blue')
plt.plot(fcast2,label="Damped",color='green')
plt.xlabel('Año')
plt.ylabel('Viajeros')
plt.title('Comparativa suavizados')
plt.legend()
plt.show()


# Aplicamos el método de Holt-Winters
model3 = ExponentialSmoothing(train, seasonal_periods=12, trend="add", 
                              seasonal="add", initialization_method="estimated").fit()
fcast3 = model3.forecast(12)
print(fcast3)

plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='yellow')
plt.plot(model3.fittedvalues, label='suavizado', color='blue')
plt.plot(fcast3,color='red', label="Prediciones")
plt.xlabel('Año')
plt.ylabel('Viajeros')
plt.title('Holt-Winter Aditivo')
plt.legend()

#Mostramos los parámetros
headers = ['Name', 'Param', 'Value', 'Optimized']
table_str = tabulate(model3.params_formatted, headers, tablefmt='fancy_grid')
print(table_str)

# Mostramos la evolución de los componentes
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
axes[0].plot(model3.level)
axes[0].set_title('Level')
axes[1].plot(model3.trend)
axes[1].set_title('Trend')
axes[2].plot(model3.season)
axes[2].set_title('Season')
plt.tight_layout()
plt.show()









