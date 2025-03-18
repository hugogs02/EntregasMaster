import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt

from tabulate import tabulate
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

warnings.simplefilter(action='ignore', category=FutureWarning)

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


# Dibujamos el correlograma
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(train, lags=20, ax=ax1)
ax1.set_title('Función de Autocorrelación (ACF) de viajeros')
plot_pacf(train, lags=20, ax=ax2)
ax2.set_title('Función de Autocorrelación Parcial (PACF) de viajeros')
plt.tight_layout()
plt.show()

# Creamos el modelo manual
modelo_arima = sm.tsa.ARIMA(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
resultados = modelo_arima.fit()
print(resultados.summary())

# Comprobamos que los residuos estén incorrelados
resultados.plot_diagnostics(figsize=(12, 8))
plt.show()

print(resultados.mse)

# Calculamos predicciones
prediciones = resultados.get_forecast(steps=12)
predi_test=prediciones.predicted_mean
print(predi_test)

# Graficamos las predicciones
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='blue')
plt.plot(prediciones.predicted_mean, label='Predicciones', color='orange')
plt.xlabel('fecha')
plt.ylabel('Viajeros')
plt.title('Modelo ARIMA')
plt.legend()
plt.show()

# Graficamos añadiendo los intervalos de confianza
intervalos_confianza = prediciones.conf_int()
plt.figure(figsize=(12, 8))
plt.plot(intervalos_confianza['lower Viajeros'], label='UCL', color='gray')
plt.plot(intervalos_confianza['upper Viajeros'], label='LCL', color='gray')
plt.plot(predi_test, label='Predicciones', color='orange')
plt.plot(test, label='Test', color='blue')
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
plt.title('Modelo ARIMA')
plt.legend()
plt.show()


# Ahora creamos el modelo automático
modelo_auto= pm.auto_arima(train, m=12, d=None, D=1, 
                           start_p=0, max_p=3, start_q=0, max_q=3,
                           start_P=0, max_P=2, start_Q=0, max_Q=2,
                           seasonal=True, trace=True,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True) 

# Imprimimos el modelo y estudiamos sus residuos
print(modelo_auto.summary())
best_arima = sm.tsa.ARIMA(train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
resultados_a = best_arima.fit()

resultados_a.plot_diagnostics(figsize=(12, 8))
plt.show()

# Calculamos las predicciones y las comparamos con los datos test
prediciones_a = resultados_a.get_forecast(steps=12)
predi_test_a=prediciones_a.predicted_mean
intervalos_confianza_a = prediciones_a.conf_int()
plt.figure(figsize=(12, 8))
plt.plot(train, label='Train', color='gray')
plt.plot(test, label='Test', color='blue')
plt.plot(prediciones_a.predicted_mean, label='Predicciones', color='orange')
plt.xlabel('Periodo')
plt.ylabel('Viajeros')
plt.title('Modelo ARIMA')
plt.legend()
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(intervalos_confianza_a['lower Viajeros'], label='UCL', color='gray')
plt.plot(intervalos_confianza_a['upper Viajeros'], label='LCL', color='gray')
plt.plot(predi_test, label='Predicciones', color='orange')
plt.plot(test, label='Test', color='blue')
plt.xlabel('Fecha')
plt.ylabel('Viajeros')
plt.title('Modelo ARIMA')
plt.legend()
plt.show()


