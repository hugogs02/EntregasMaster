# Cargo las librerias 
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
os.chdir('C:\DocumentaciónMineria_Python\Datos')

# Cargo las funciones que voy a utilizar despues
from FuncionesMineria import (graficoVcramer, impVariablesLog, pseudoR2, glm, summary_glm, 
                           validacion_cruzada_glm, sensEspCorte, crear_data_modelo, curva_roc)

# Cargo los datos depurados (incluidas las mejores transformaciones de las 
# variables numericas respecto a la binaria)
with open('todo_bin.pickle', 'rb') as f:
    todo = pickle.load(f)
    
# Identifico la variable objetivo y la elimino de mi conjunto de datos.
varObjBin = todo['Compra']
todo = todo.drop('Compra', axis = 1)

# Veo el reparto original. Compruebo que la variable objetivo tome valor 1 para el evento y 0 para el no evento
pd.DataFrame({
    'n': varObjBin.value_counts()
    , '%': varObjBin.value_counts(normalize = True)
})

# Pruebo un primer modelo con las variables originales
eliminar = ['xAcidez', 'xAcidoCitrico', 'xAzucar', 'xCloruroSodico', 'xDensidad', 'xpH', 
            'xSulfatos', 'xAlcohol', 'xPrecioBotella']
todo = todo.drop(eliminar, axis = 1)

# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(todo, varObjBin, test_size = 0.2, random_state = 1234567)
# Indico que la variable respuesta es numérica (hay que introducirla en el algoritmo de phython tal y como la va a tratar)
y_train, y_test = y_train.astype(int), y_test.astype(int)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
var_cont1 = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 'Sulfatos', 
             'Alcohol', 'PrecioBotella']
var_categ1 = ['Etiqueta', 'CalifProductor', 'Clasificacion', 'Region', 'prop_missings']

# Creo el modelo inicial
modeloInicial = glm(y_train, x_train, var_cont1, var_categ1)
# Visualizamos los resultado del modelo
summary_glm(modeloInicial['Modelo'], y_train, modeloInicial['X'])

# Calculamos la medida de ajuste R^2 para los datos de entrenamiento
pseudoR2(modeloInicial['Modelo'], modeloInicial['X'], y_train)

# Preparamos los datos test para usar en el modelo
x_test_modeloInicial = crear_data_modelo(x_test, var_cont1, var_categ1)

# Calculamos la medida de ajuste R^2 para los datos test
pseudoR2(modeloInicial['Modelo'], x_test_modeloInicial, y_test)

# Calculamos el número de parámetros utilizados en el modelo.
len(modeloInicial['Modelo'].coef_[0])

# Fijandome en la significacion de las variables, el modelo con las variables mas significativas queda
var_cont2 = ['CloruroSodico', 'Sulfatos', 'pH']
var_categ2 = ['Etiqueta', 'CalifProductor', 'Clasificacion', 'prop_missings']

modelo2 = glm(y_train, x_train, var_cont2, var_categ2)

summary_glm(modelo2['Modelo'], y_train, modelo2['X'])

pseudoR2(modelo2['Modelo'], modelo2['X'], y_train)

x_test_modelo2 = crear_data_modelo(x_test, var_cont2, var_categ2)
pseudoR2(modelo2['Modelo'], x_test_modelo2, y_test)

len(modelo2['Modelo'].coef_[0])

# Calculamos y representamos la importancia de las variables en el modelo
impVariablesLog(modelo2, y_train, x_train, var_cont2, var_categ2)

# Calculamos el area bajo la curva ROC y representamos
AUC2 = curva_roc(x_test_modelo2, y_test, modelo2)

# Miro el grafico V de Cramer para ver las variables mas importantes
graficoVcramer(todo, varObjBin) 

var_cont3 = ['Densidad', 'Acidez', 'CloruroSodico']
var_categ3 = ['CalifProductor', 'Clasificacion', 'prop_missings']

modelo3 = glm(y_train, x_train, var_cont3, var_categ3)

summary_glm(modelo3['Modelo'], y_train, modelo3['X'])

pseudoR2(modelo3['Modelo'], modelo3['X'], y_train)

x_test_modelo3 = crear_data_modelo(x_test, var_cont3, var_categ3)
pseudoR2(modelo3['Modelo'], x_test_modelo3, y_test)

len(modelo3['Modelo'].coef_[0])


# Pruebo alguna interaccion sobre el modelo 3
var_cont4 = var_cont3
var_categ4 = var_categ3
var_interac4 = [('Clasificacion', 'CalifProductor')]
modelo4 = glm(y_train, x_train, var_cont4, var_categ4, var_interac4)
summary_glm(modelo4['Modelo'], y_train, modelo4['X'])
pseudoR2(modelo4['Modelo'], modelo4['X'], y_train)
x_test_modelo4 = crear_data_modelo(x_test, var_cont4, var_categ4, var_interac4)
pseudoR2(modelo4['Modelo'], x_test_modelo4, y_test)
len(modelo4['Modelo'].coef_[0])



# Pruebo uno con las variables mas importantes del 2 
var_cont5 = []
var_categ5 = ['Clasificacion', 'CalifProductor', 'Etiqueta']
modelo5 = glm(y_train, x_train, var_cont5, var_categ5)
summary_glm(modelo5['Modelo'], y_train, modelo5['X'])
pseudoR2(modelo5['Modelo'], modelo5['X'], y_train)
x_test_modelo5 = crear_data_modelo(x_test, var_cont5, var_categ5)
pseudoR2(modelo5['Modelo'], x_test_modelo5, y_test)
len(modelo5['Modelo'].coef_[0])

# Pruebo uno con las variables mas importantes del 2 y una interaccion
var_cont6 = []
var_categ6 = ['Clasificacion', 'CalifProductor', 'Etiqueta']
var_interac6 = [('Clasificacion', 'Etiqueta')]
modelo6 = glm(y_train, x_train, var_cont6, var_categ6, var_interac6)
summary_glm(modelo6['Modelo'], y_train, modelo6['X'])
pseudoR2(modelo6['Modelo'], modelo6['X'], y_train)
x_test_modelo6 = crear_data_modelo(x_test, var_cont6, var_categ6, var_interac6)
pseudoR2(modelo6['Modelo'], x_test_modelo6, y_test)
len(modelo6['Modelo'].coef_[0])

# Mejor modelo según el Área bajo la Curva ROC
AUC1 = curva_roc(x_test_modeloInicial, y_test, modeloInicial)
AUC2 = curva_roc(x_test_modelo2, y_test, modelo2)
AUC3 = curva_roc(x_test_modelo3, y_test, modelo3)
AUC4 = curva_roc(x_test_modelo4, y_test, modelo4)
AUC5 = curva_roc(x_test_modelo5, y_test, modelo5)
AUC6 = curva_roc(x_test_modelo6, y_test, modelo6)

# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_glm(5, x_train, y_train, var_cont1, var_categ1)
    modelo2VC = validacion_cruzada_glm(5, x_train, y_train, var_cont2, var_categ2)
    modelo3VC = validacion_cruzada_glm(5, x_train, y_train, var_cont3, var_categ3)
    modelo4VC = validacion_cruzada_glm(5, x_train, y_train, var_cont4, var_categ4, var_interac4)
    modelo5VC = validacion_cruzada_glm(5, x_train, y_train, var_cont5, var_categ5)
    modelo6VC = validacion_cruzada_glm(5, x_train, y_train, var_cont6, var_categ6, var_interac6)
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'AUC': modelo1VC + modelo2VC + modelo3VC + modelo4VC + modelo5VC + modelo6VC
        , 'Resample': ['Rep' + str((rep + 1))]*5*6  # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)

# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de AUC por modelo
grupo_metrica = results.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

 
    
# Calcular la media del AUC por modelo
results.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar del AUC por modelo
results.groupby('Modelo')['AUC'].std()    
# Contar el número de parámetros en cada modelo
num_params = [len(modeloInicial['Modelo'].coef_[0]), len(modelo2['Modelo'].coef_[0]), len(modelo3['Modelo'].coef_[0]), 
 len(modelo4['Modelo'].coef_[0]), len(modelo5['Modelo'].coef_[0]), len(modelo6['Modelo'].coef_[0])]

print(num_params)

## Buscamos el mejor punto de corte

# Probamos dos
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.4, var_cont5, var_categ5)
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.6, var_cont5, var_categ5)

# Generamos una rejilla de puntos de corte
posiblesCortes = np.arange(0, 1.01, 0.01).tolist()  # Generamos puntos de corte de 0 a 1 con intervalo de 0.01
rejilla = pd.DataFrame({
    'PtoCorte': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'PosPredValue': [],
    'NegPredValue': []
})  # Creamos un DataFrame para almacenar las métricas para cada punto de corte

for pto_corte in posiblesCortes:  # Iteramos sobre los puntos de corte
    rejilla = pd.concat(
        [rejilla, sensEspCorte(modelo5['Modelo'], x_test, y_test, pto_corte, var_cont5, var_categ5)],
        axis=0
    )  # Calculamos las métricas para el punto de corte actual y lo agregamos al DataFrame

rejilla['Youden'] = rejilla['Sensitivity'] + rejilla['Specificity'] - 1  # Calculamos el índice de Youden
rejilla.index = list(range(len(rejilla)))  # Reindexamos el DataFrame para que los índices sean consecutivos



plt.plot(rejilla['PtoCorte'], rejilla['Youden'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Youden')
plt.title('Youden')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Accuracy'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

rejilla['PtoCorte'][rejilla['Youden'].idxmax()]
rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()]

# El resultado es 0.75 para youden y 0.5 para Accuracy
# Los comparamos
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.75, var_cont5, var_categ5)
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.5, var_cont5, var_categ5)

# Vemos las variables mas importantes del modelo ganador
impVariablesLog(modelo5, y_train, x_train, var_cont5, var_categ5)

# Vemos los coeficientes del modelo ganador
coeficientes = modelo5['Modelo'].coef_
nombres_caracteristicas = crear_data_modelo(x_train, var_cont5, var_categ5).columns  # Suponiendo que X_train es un DataFrame de pandas
# Imprime los nombres de las características junto con sus coeficientes
for nombre, coef in zip(nombres_caracteristicas, coeficientes[0]):
    print(f"Variable: {nombre}, Coeficiente: {coef}")

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
pseudoR2(modelo5['Modelo'], modelo5['X'], y_train)
pseudoR2(modelo5['Modelo'], x_test_modelo5, y_test)
# Es poca la diferencia, por lo que el modelo se puede considerar robusto

# Calculamos la diferencia del Area bajo la curva ROC en train y test
curva_roc(crear_data_modelo(x_train, var_cont5, var_categ5), y_train, modelo5)
curva_roc(x_test_modelo5, y_test, modelo5)

# Calculamos la diferencia de las medidas de calidad entre train y test 
sensEspCorte(modelo5['Modelo'], x_train, y_train, 0.5, var_cont5, var_categ5)
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.5, var_cont5, var_categ5)
