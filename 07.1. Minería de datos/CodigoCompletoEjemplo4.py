# Cargo las librerias 
import os
import pickle
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


os.chdir('C:\DocumentaciónMineria_Python\Datos')

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (Rsq, lm, lm_forward, lm_backward, lm_stepwise, validacion_cruzada_lm,
                           crear_data_modelo, modelEffectSizes)

# Cargo los datos depurados
with open('todo_cont.pickle', 'rb') as f:
    todo = pickle.load(f)

# Identifico la variable objetivo y la elimino del conjunto de datos
varObjCont = todo['Beneficio']
todo = todo.drop('Beneficio', axis = 1)

# Identifico las variables continuas
var_cont = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 
            'Sulfatos', 'Alcohol',  'PrecioBotella', 'xAcidez', 
            'cuartaxAcidoCitrico', 'sqrtxAzucar', 'xCloruroSodico', 'xDensidad', 
            'sqrxpH', 'logxSulfatos', 'xAlcohol', 'xPrecioBotella']

# Identifico las variables continuas sin transformar
var_cont_sin_transf = ['Acidez', 'AcidoCitrico', 'Azucar', 'CloruroSodico', 'Densidad', 'pH', 
                       'Sulfatos', 'Alcohol',  'PrecioBotella']

# Identifico las variables categóricas
var_categ = ['Etiqueta', 'CalifProductor', 'Clasificacion', 'Region', 'prop_missings']

# Hago la particion
x_train, x_test, y_train, y_test = train_test_split(todo, varObjCont, test_size = 0.2, random_state = 1234567)

#y_train, y_test = y_train.astype(int), y_test.astype(int)

# Construyo el modelo ganador del dia 2
modeloManual = lm(y_train, x_train, [], ['Clasificacion', 'Etiqueta', 'CalifProductor'])
# Resumen del modelo
modeloManual['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloManual['Modelo'], y_train, modeloManual['X'])
# Preparo datos test
x_test_modeloManual = crear_data_modelo(x_test, [], ['Clasificacion', 'Etiqueta', 'CalifProductor'])
# R-squared del modelo para test
Rsq(modeloManual['Modelo'], y_test, x_test_modeloManual)

# Como la categoría 2 de la varaibles CalifProductor no es significativa
# la unimos con la categoría '0-1' 

todo['CalifProductor2'] = todo['CalifProductor'].replace({'2': '0-1'})

# Hago de nuevo la partición porque hay una nueva variable en el conjunto de datos "Todo"
x_train, x_test, y_train, y_test = train_test_split(todo, varObjCont, test_size = 0.2, random_state = 1234567)


# Construyo el modelo ganador del día 2 con la nueva variable CalifProductor2
modeloManual2 = lm(y_train, x_train, [], ['Clasificacion', 'Etiqueta', 'CalifProductor2'])

# Resumen del modelo
modeloManual2['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloManual2['Modelo'], y_train, modeloManual2['X'])
# Preparo datos test
x_test_modeloManual2 = crear_data_modelo(x_test, [], ['Clasificacion', 'Etiqueta', 'CalifProductor2'])
# R-squared del modelo para test
Rsq(modeloManual2['Modelo'], y_test, x_test_modeloManual2)

# Modelo muy similar R^2 similares con CalifProductor y CalifProductor2
# Con CalifProductor2 todos los parámetros son significativos.

# Añadimos a las variables categóricas la variable CalifProductor2

var_categ = ['Etiqueta', 'CalifProductor', 'CalifProductor2', 'Clasificacion', 'Region', 'prop_missings']


# Seleccion de variables Stepwise, métrica AIC
modeloStepAIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])

# Preparo datos test
x_test_modeloStepAIC = crear_data_modelo(x_test, modeloStepAIC['Variables']['cont'], 
                                                modeloStepAIC['Variables']['categ'], 
                                                modeloStepAIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)

# Seleccion de variables Backward, métrica AIC
modeloBackAIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'AIC')
modeloBackAIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])


x_test_modeloBackAIC = crear_data_modelo(x_test, modeloBackAIC['Variables']['cont'], 
                                                modeloBackAIC['Variables']['categ'], 
                                                modeloBackAIC['Variables']['inter'])


Rsq(modeloBackAIC['Modelo'], y_test, x_test_modeloBackAIC)

# Comparo número de parámetros (iguales)
len(modeloStepAIC['Modelo'].params)
len(modeloBackAIC['Modelo'].params)


# Mismas variables seleccionadas, mismos parámetros, mismo modelo.
modeloStepAIC['Modelo'].params
modeloBackAIC['Modelo'].params

# Seleccion de variables Stepwise, métrica BIC
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])

# Preparo datos test
x_test_modeloStepBIC = crear_data_modelo(x_test, modeloStepBIC['Variables']['cont'], 
                                                modeloStepBIC['Variables']['categ'], 
                                                modeloStepBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

# Seleccion de variables Backward, métrica BIC
modeloBackBIC = lm_backward(y_train, x_train, var_cont_sin_transf, var_categ, [], 'BIC')
# Resumen del modelo
modeloBackBIC['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])

# Preparo datos test
x_test_modeloBackBIC = crear_data_modelo(x_test, modeloBackBIC['Variables']['cont'], 
                                                modeloBackBIC['Variables']['categ'], 
                                                modeloBackBIC['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloBackBIC['Modelo'], y_test, x_test_modeloBackBIC)

# Comparo número de parámetros
len(modeloBackBIC['Modelo'].params)
len(modeloStepBIC['Modelo'].params)


# Mismas variables seleccionadas, mismos parámetros, mismo modelo.
# Los metodos Stepwise y Backward han resultado ser iguales. Pero distintos a los del
# modeloStepAIC
modeloStepBIC['Modelo'].params
modeloBackBIC['Modelo'].params


# Comparo (R-squared)
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])

Rsq(modeloStepAIC['Modelo'], y_test, x_test_modeloStepAIC)
Rsq(modeloStepBIC['Modelo'], y_test, x_test_modeloStepBIC)

# Nos quedamos con modeloStepBIC=modeloBackBIC, tienen similar R-squared pero menos parámetros


# Interacciones 2 a 2 de todas las variables (excepto las continuas transformadas)
interacciones = var_cont_sin_transf + var_categ
interacciones_unicas = list(itertools.combinations(interacciones, 2)) 
  
# Seleccion de variables Stepwise, métrica AIC, con interacciones
modeloStepAIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ, 
                                interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_int['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])

# Preparo datos test
x_test_modeloStepAIC_int = crear_data_modelo(x_test, modeloStepAIC_int['Variables']['cont'], 
                                                    modeloStepAIC_int['Variables']['categ'], 
                                                    modeloStepAIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)

# Seleccion de variables Stepwise, métrica BIC, con interacciones
modeloStepBIC_int = lm_stepwise(y_train, x_train, var_cont_sin_transf, var_categ,
                                interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_int['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])

# Preparo datos test
x_test_modeloStepBIC_int = crear_data_modelo(x_test, modeloStepBIC_int['Variables']['cont'], 
                                                    modeloStepBIC_int['Variables']['categ'], 
                                                    modeloStepBIC_int['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)
  


# Comparo los R^2 del modelo utilizando ambos criterios
Rsq(modeloStepAIC_int['Modelo'], y_train, modeloStepAIC_int['X'])
Rsq(modeloStepAIC_int['Modelo'], y_test, x_test_modeloStepAIC_int)


Rsq(modeloStepBIC_int['Modelo'], y_train, modeloStepBIC_int['X'])
Rsq(modeloStepBIC_int['Modelo'], y_test, x_test_modeloStepBIC_int)



# Comparo número de parámetros  
# Por el principio de parsimonia, es preferible el modeloStepBIC_int 
len(modeloStepAIC_int['Modelo'].params)
len(modeloStepBIC_int['Modelo'].params)


# Pruebo con todas las transf y las variables originales, métrica AIC
modeloStepAIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'AIC')
# Resumen del modelo
modeloStepAIC_trans['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])

# Preparo datos test
x_test_modeloStepAIC_trans = crear_data_modelo(x_test, modeloStepAIC_trans['Variables']['cont'], 
                                                      modeloStepAIC_trans['Variables']['categ'], 
                                                      modeloStepAIC_trans['Variables']['inter'])

# R-squared del modelo para test
Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)

# Pruebo con todas las transf y las variables originales, métrica BIC
modeloStepBIC_trans = lm_stepwise(y_train, x_train, var_cont, var_categ, [], 'BIC')
# Resumen del modelo
modeloStepBIC_trans['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])

# Preparo datos test
x_test_modeloStepBIC_trans = crear_data_modelo(x_test, modeloStepBIC_trans['Variables']['cont'], 
                                                      modeloStepBIC_trans['Variables']['categ'], 
                                                      modeloStepBIC_trans['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)


# Comparo los R^2 de los modelos utilizando ambos criterios
Rsq(modeloStepAIC_trans['Modelo'], y_train, modeloStepAIC_trans['X'])
Rsq(modeloStepAIC_trans['Modelo'], y_test, x_test_modeloStepAIC_trans)


Rsq(modeloStepBIC_trans['Modelo'], y_train, modeloStepBIC_trans['X'])
Rsq(modeloStepBIC_trans['Modelo'], y_test, x_test_modeloStepBIC_trans)


# Comparo número de parámetros  
# No está claro cual es mejor
# Nos quedamos con los dos para decirlo con validación cruzada
len(modeloStepAIC_trans['Modelo'].params)
len(modeloStepBIC_trans['Modelo'].params)

# Pruebo modelo con las Transformaciones y las interacciones, métrica AIC
modeloStepAIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
# Resumen del modelo
modeloStepAIC_transInt['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepAIC_transInt['Modelo'], y_train, modeloStepAIC_transInt['X'])

# Preparo datos test
x_test_modeloStepAIC_transInt = crear_data_modelo(x_test, modeloStepAIC_transInt['Variables']['cont'], 
                                                         modeloStepAIC_transInt['Variables']['categ'], 
                                                         modeloStepAIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepAIC_transInt['Modelo'], y_test, x_test_modeloStepAIC_transInt)

# Pruebo modelo con las Transformaciones y las interacciones, métrica BIC
modeloStepBIC_transInt = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
# Resumen del modelo
modeloStepBIC_transInt['Modelo'].summary()

# R-squared del modelo para train
Rsq(modeloStepBIC_transInt['Modelo'], y_train, modeloStepBIC_transInt['X'])


# Preparo datos test
x_test_modeloStepBIC_transInt = crear_data_modelo(x_test, modeloStepBIC_transInt['Variables']['cont'], 
                                                         modeloStepBIC_transInt['Variables']['categ'], 
                                                         modeloStepBIC_transInt['Variables']['inter'])
# R-squared del modelo para test
Rsq(modeloStepBIC_transInt['Modelo'], y_test, x_test_modeloStepBIC_transInt)


# Comparo los R^2 de los modelos utilizando ambos criterios
Rsq(modeloStepAIC_transInt['Modelo'], y_train, modeloStepAIC_transInt['X'])
Rsq(modeloStepAIC_transInt['Modelo'], y_test, x_test_modeloStepAIC_transInt)


Rsq(modeloStepBIC_transInt['Modelo'], y_train, modeloStepBIC_transInt['X'])
Rsq(modeloStepBIC_transInt['Modelo'], y_test, x_test_modeloStepBIC_transInt)


# Comparo número de parámetros  
# Por el principio de parsimonia, es preferible el modeloStepBIC_transInt
len(modeloStepAIC_transInt['Modelo'].params)
len(modeloStepBIC_transInt['Modelo'].params)


# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)

for rep in range(20):
    # Realiza validación cruzada en seis modelos diferentes y almacena sus R-squared en listas separadas

    modelo_manual2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloManual2['Variables']['cont']
        , modeloManual2['Variables']['categ']
    )
    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC['Variables']['cont']
        , modeloStepBIC['Variables']['categ']
    )
    modelo_stepBIC_int = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_int['Variables']['cont']
        , modeloStepBIC_int['Variables']['categ']
        , modeloStepBIC_int['Variables']['inter']
    )
    modelo_stepAIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepAIC_trans['Variables']['cont']
        , modeloStepAIC_trans['Variables']['categ']
    )
    modelo_stepBIC_trans = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
    )
    modelo_stepBIC_transInt = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_transInt['Variables']['cont']
        , modeloStepBIC_transInt['Variables']['categ']
        , modeloStepBIC_transInt['Variables']['inter']
    )
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición

    results_rep = pd.DataFrame({
        'Rsquared': modelo_manual2 + modelo_stepBIC + modelo_stepBIC_int + modelo_stepBIC_trans + modelo_stepBIC_trans + modelo_stepBIC_transInt
        , 'Resample': ['Rep' + str((rep + 1))]*5*6 # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)
    
# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Mayor R^2 Modelo 3 (stepBIC_int) y Modelo 6 (stepBIC_transInt), 
# pero también los que presentan mayor variabilidad

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print(media_r2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)
# Contar el número de parámetros en cada modelo
num_params = [len(modeloManual2['Modelo'].params), len(modeloStepAIC['Modelo'].params), len(modeloStepBIC_int['Modelo'].params), 
 len(modeloStepAIC_trans['Modelo'].params), len(modeloStepBIC_trans['Modelo'].params), 
 len(modeloStepBIC_transInt['Modelo'].params)]

print(num_params)

# La mejora en R^2 para el modelo 3 (modeloStepBIC_int) y modelo 6 (modeloStepBIC_transInt)
# junto con el elevado número de parámetros que estos modelos presentan, y la mayor variabilidad
# hace que los descartemos como modelos ganadores.
# Elejimos el modelo 5 (modeloStepBIC_trans) por su reducido número de parámetros.
# Tiene una variabilidad más pequeña que el modelo 1.

## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)
# Concretamente el 70% de los datos de entrenamiento utilizados para contruir los 
# modelos anteriores.
# El método de selección usado ha sido el Stepwise con el criterio BIC
# Se aplica este método a 30 submuestras diferentes

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 1234567 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
    
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]

# Si quisiéramos mostrar los tres modelos más frecuentes añadiríamos la siguiente línea de código
# var_3 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
#    frec_ordenada['Formula'][2])]



## Comparacion final, tomo el ganador de antes y los nuevos candidatos
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_trans['Variables']['cont']
        , modeloStepBIC_trans['Variables']['categ']
        , modeloStepBIC_trans['Variables']['inter']
    )
    modelo2 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_1['cont']
        , var_1['categ']
        , var_1['inter']
    )
    modelo3 = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , var_2['cont']
        , var_2['categ']
        , var_2['inter']
    )
    
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 
        , 'Resample': ['Rep' + str((rep + 1))]*5*3
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 
    })
    results = pd.concat([results, results_rep], axis = 0)

# Boxplot de la validacion cruzada 
     
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

# Modelo 2 y 3 muestran un mayor valor medio de R^2 que el modelo 1 pero mayor variabilidad
# Observamos esto numéricamente así como el número de parámetros de cada modelo para elegir el ganador

# Calcular la media de las métricas R-squared por modelo
media_r2_v2 = results.groupby('Modelo')['Rsquared'].mean()
print (media_r2_v2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2_v2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2_v2)
# Contar el número de parámetros en cada modelo
num_params_v2 = [len(modeloStepBIC_trans['Modelo'].params), 
                 len(frec_ordenada['Formula'][0].split('+')),
                 len(frec_ordenada['Formula'][1].split('+'))]

print(num_params_v2)

# El aumento en el número del parámetros de los modelos 2 y 3 no justifica
# el aumento en el R^2, por lo que elegimos como modelo ganador el modelo 1 (modeloStepBIC_trans)
# Este es el modelo elegido utilizando los métodos de selección clásica 
# Una vez decidido el mejor modelo, hay que evaluarlo 
ModeloGanador = modeloStepBIC_trans

# Vemos los coeficientes del modelo ganador
ModeloGanador['Modelo'].summary()
# Todos los parámetros del modelo son significativos

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test
Rsq(ModeloGanador['Modelo'], y_train, ModeloGanador['X'])

x_test_modeloganador = crear_data_modelo(x_test, ModeloGanador['Variables']['cont'], 
                                                ModeloGanador['Variables']['categ'], 
                                                ModeloGanador['Variables']['inter'])
Rsq(ModeloGanador['Modelo'], y_test, x_test_modeloganador)
    
modelEffectSizes(ModeloGanador, y_train, x_train, ModeloGanador['Variables']['cont'], 
                  ModeloGanador['Variables']['categ'], ModeloGanador['Variables']['inter'])
   
    
