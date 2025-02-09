# Cargo las librerias 
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

os.chdir('C:\DocumentaciónMineria_Python\Datos')

# Cargo las funciones que voy a utilizar despues
from FuncionesMineria import (graficoVcramer, mosaico_targetbinaria, boxplot_targetbinaria, 
                           hist_targetbinaria, Transf_Auto, lm, Rsq, validacion_cruzada_lm,
                           modelEffectSizes, crear_data_modelo, Vcramer)

# Parto de los datos ya depurados
with open('datosVinoDep.pickle', 'rb') as f:
    datos = pickle.load(f)

# Defino las variables objetivo y las elimino del conjunto de datos input
varObjCont = datos['Beneficio']
varObjBin = datos['Compra']
datos_input = datos.drop(['Beneficio', 'Compra'], axis = 1) 
 
# Genera una lista con los nombres de las variables.
variables = list(datos_input.columns)  

# Obtengo la importancia de las variables
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)

# Crear un DataFrame para almacenar los resultados del coeficiente V de Cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjCont)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjCont.name, 'Vcramer': v_cramer},
                             ignore_index=True)
    
for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjBin)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjBin.name, 'Vcramer': v_cramer},
                             ignore_index=True)

# Veo graficamente el efecto de dos variables cualitativas sobre la binaria
# Tomo las variables con más y menos relación con la variable objetivo Binaria
mosaico_targetbinaria(datos_input['Region'], varObjBin, 'Region')
mosaico_targetbinaria(datos_input['Clasificacion'], varObjBin, 'Clasificacion')

# Veo graficamente el efecto de dos variables cuantitativas sobre la binaria
boxplot_targetbinaria(datos_input['Alcohol'], varObjBin,'Objetivo', 'Alcohol')
boxplot_targetbinaria(datos_input['CloruroSodico'], varObjBin, 'Objetivo','CloruroSodico')

hist_targetbinaria(datos_input['Alcohol'], varObjBin, 'Alcohol')
hist_targetbinaria(datos_input['CloruroSodico'], varObjBin, 'CloruroSodico')

# Correlación entre todas las variables numéricas frente a la objetivo continua.
# Obtener las columnas numéricas del DataFrame 'datos_input'
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
# Calcular la matriz de correlación de Pearson entre la variable objetivo continua ('varObjCont') y las variables numéricas
matriz_corr = pd.concat([varObjCont, datos_input[numericas]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
plt.figure(figsize=(8, 6))
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Crear un mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
# Establecer el título del gráfico
plt.title("Matriz de correlación")
# Mostrar el gráfico de la matriz de correlación
plt.show()

# Busco las mejores transformaciones para las variables numericas con respesto a los dos tipos de variables
input_cont = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjCont)], axis = 1)
input_bin = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjBin)], axis = 1)

# Creamos conjuntos de datos que contengan las variables explicativas y una de las variables objetivo y los guardamos
todo_cont = pd.concat([input_cont, varObjCont], axis = 1)
todo_bin = pd.concat([input_bin, varObjBin], axis = 1)
with open('todo_bin.pickle', 'wb') as archivo:
    pickle.dump(todo_bin, archivo)
with open('todo_cont.pickle', 'wb') as archivo:
    pickle.dump(todo_cont, archivo)


## Comenzamos con la regresion lineal

# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos_input, np.ravel(varObjCont), test_size = 0.2, random_state = 123456)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
var_cont1 = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 'Sulfatos', 
             'Alcohol', 'PrecioBotella']
var_categ1 = ['Etiqueta', 'CalifProductor', 'Clasificacion', 'Region', 'prop_missings']

# Creo el modelo
modelo1 = lm(y_train, x_train, var_cont1, var_categ1)
# Visualizamos los resultado del modelo
modelo1['Modelo'].summary()

# Calculamos la medida de ajuste R^2 para los datos de entrenamiento
Rsq(modelo1['Modelo'], y_train, modelo1['X'])

# Preparamos los datos test para usar en el modelo
x_test_modelo1 = crear_data_modelo(x_test, var_cont1, var_categ1)
# Calculamos la medida de ajuste R^2 para los datos test
Rsq(modelo1['Modelo'], y_test, x_test_modelo1)


# Nos fijamos en la importancia de las variables
modelEffectSizes(modelo1, y_train, x_train, var_cont1, var_categ1)


# Vamos a probar un modelo con menos variables. Recuerdo el grafico de Cramer
graficoVcramer(datos_input, varObjCont) # Pruebo con las mas importantes

# Construyo el segundo modelo
var_cont2 = []
var_categ2 = ['Etiqueta', 'CalifProductor', 'Clasificacion','prop_missings']
modelo2 = lm(y_train, x_train, var_cont2, var_categ2)
modelEffectSizes(modelo2, y_train, x_train, var_cont2, var_categ2)
modelo2['Modelo'].summary()
Rsq(modelo2['Modelo'], y_train, modelo2['X'])
x_test_modelo2 = crear_data_modelo(x_test, var_cont2, var_categ2)
Rsq(modelo2['Modelo'], y_test, x_test_modelo2)

# Pruebo un modelo con menos variables, basandome en la importancia de las variables
var_cont3 = []
var_categ3 = ['Etiqueta', 'CalifProductor', 'Clasificacion']
modelo3 = lm(y_train, x_train, var_cont3, var_categ3)
modelo3['Modelo'].summary()
Rsq(modelo3['Modelo'], y_train, modelo3['X'])
x_test_modelo3 = crear_data_modelo(x_test, var_cont3, var_categ3)
Rsq(modelo3['Modelo'], y_test, x_test_modelo3)

# Pruebo con una interaccion sobre el anterior
# Se podrian probar todas las interacciones dos a dos
var_cont4 = []
var_categ4 = ['Etiqueta', 'CalifProductor', 'Clasificacion']
var_interac4 = [('Clasificacion', 'Etiqueta')]
modelo4 = lm(y_train, x_train, var_cont4, var_categ4, var_interac4)
modelo4['Modelo'].summary()
Rsq(modelo4['Modelo'], y_train, modelo4['X'])
x_test_modelo4 = crear_data_modelo(x_test, var_cont4, var_categ4, var_interac4)
Rsq(modelo4['Modelo'], y_test, x_test_modelo4)


# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': [],
    'Resample': [],
    'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_lm(5, x_train, y_train, var_cont1, var_categ1)
    modelo2VC = validacion_cruzada_lm(5, x_train, y_train, var_cont2, var_categ2)
    modelo3VC = validacion_cruzada_lm(5, x_train, y_train, var_cont3, var_categ3)
    modelo4VC = validacion_cruzada_lm(5, x_train, y_train, var_cont4, var_categ4, var_interac4)
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'Rsquared': modelo1VC + modelo2VC + modelo3VC + modelo4VC,
        'Resample': ['Rep' + str((rep + 1))] * 5 * 4,  # Etiqueta de repetición
        'Modelo': [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5  # Etiqueta de modelo (1, 2, 3 o 4)
    })
    
    # Concatena los resultados de esta repetición al DataFrame principal 'results'
    results = pd.concat([results, results_rep], axis=0)

    
# Boxplot de la validación cruzada
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de R-squared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico 
    

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print(media_r2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)
# Contar el número de parámetros en cada modelo
num_params = [len(modelo1['Modelo'].params), len(modelo2['Modelo'].params), 
             len(modelo3['Modelo'].params), len(modelo4['Modelo'].params)]

# Teniendo en cuenta el R2, la estabilidad y el numero de parametros, nos quedamos con el modelo3
# Vemos los coeficientes del modelo ganador
modelo3['Modelo'].summary()

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
Rsq(modelo3['Modelo'], y_train, modelo3['X'])
Rsq(modelo3['Modelo'], y_test, x_test_modelo3)

# Vemos las variables mas importantes del modelo ganador

modelEffectSizes(modelo3, y_train, x_train, var_cont3, var_categ3)
