import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import itertools
from sklearn.model_selection import train_test_split
from collections import Counter
os.chdir("C:\\Users\\Hugo\\OneDrive\\Desktop\\Entregas\\EntregasMaster\\07.1. Minería de datos")
from FuncionesMineria import *


# Leemos los datos, los mostramos y mostramos sus tipos
datos = pd.read_excel("DatosElecciones.xlsx", sheet_name='DatosEleccionesEspaña')
print(datos.head(5))
print(datos[['Izquierda', 'Derecha', 'AbstencionAlta']].dtypes)

# Dividimos las variables en categóricas y numéricas, y comprobamos que tengan el tipo correcto.
variables = list(datos.columns)
numsACategoricas = ['Izquierda', 'Derecha', 'AbstencionAlta']
for v in numsACategoricas:
    datos[v] = datos[v].astype(str)
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns
categoricas = [v for v in variables if v not in numericas]
print(categoricas)

# Analizamos la frecuencia de los valores en las categóricas, así como sus valores únicos
print(analizar_variables_categoricas(datos))

for cat in categoricas:
    print(datos[cat].unique())

# Analizamos las variables numéricas, contando los valores únicos y comprobando los decriptivos
print(cuentaDistintos(datos))

descriptivos_num = datos.describe().T
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)

print(descriptivos_num)

erroresCenso = datos.loc[datos["TotalCensus"] > datos["Population"]]
print(erroresCenso[['Name', 'TotalCensus', 'Population']])

# Comprobamos si hay valores faltantes
print(datos[variables].isna().sum())

# Corregimos los errores

# Categoricas
# Agrupamos construccion e industria en otro
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Construccion': 'Otro', 'Industria': 'Otro'})

# Supongamos que tu DataFrame se llama df y tiene las columnas 'CCAA' y 'abstentionptge'
promedio_abstencion = datos.groupby('CCAA')['AbstentionPtge'].mean().reset_index()
promedio_abstencion.sort_values(by='AbstentionPtge')

# Transformamos CCAA
datos['CCAA'] = datos['CCAA'].replace({'Canarias': 'Can-Ast-Bal-PV', 'Asturias': 'Can-Ast-Bal-PV',
                                       'Baleares': 'Can-Ast-Bal-PV', 'PaísVasco': 'Can-Ast-Bal-PV',
                                       'Galicia': 'Galicia-Navarra', 'Navarra': 'Galicia-Navarra',
                                       'Murcia': 'Mur-Can-Ext', 'Cantabria': 'Mur-Can-Ext', 'Extremadura': 'Mur-Can-Ext',
                                       'Madrid': 'Madrid-Aragón', 'Aragón': 'Madrid-Aragón',
                                       'Rioja': 'Rioja-ComValenciana', 'ComValenciana': 'Rioja-ComValenciana'})
# Revisamos de nuevo las frecuencias
print(analizar_variables_categoricas(datos)['CCAA'])

for v in categoricas:
    datos[v] = datos[v].replace('nan', np.nan)
    
datos['Explotaciones'] = datos['Explotaciones'].replace(99999,np.nan)

# Numericas
# Transformamos población
"""datos["Population"] = np.log1p(datos["Population"])
datos["Pob2010"] = np.log1p(datos["Pob2010"])
datos["inmuebles"] = np.log1p(datos["inmuebles"])
datos["TotalCensus"] = np.log1p(datos["TotalCensus"])"""

# Eliminamos valores fuera de rango en los porcentajes
datos['Age_over65_pct'] = [x if 0<=x<=100 else np.nan for x in datos['Age_over65_pct']]
datos['ForeignersPtge'] = [x if 0<=x<=100 else np.nan for x in datos['ForeignersPtge']]
datos['Age_19_65_pct'] = [x if 0<=x<=100 else np.nan for x in datos['Age_19_65_pct']]
datos['SameComAutonPtge'] = [x if 0<=x<=100 else np.nan for x in datos['SameComAutonPtge']]
newdes=datos.describe().T

# Indicamos ID
datos.set_index(["Name", "CodigoProvincia"], inplace=True, drop=True)

# Definimos variables
varObjCont = datos['AbstentionPtge']
varObjBin = datos['Izquierda']

# Eliminamos todas las variables objetivo
datos_input = datos.drop(['AbstencionAlta', 'AbstentionPtge', 'Izda_Pct', 'Dcha_Pct',
                          'Otros_Pct', 'Izquierda', 'Derecha'], axis=1)

variables_input = list(datos_input.columns)  
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns
categoricas_input = [v for v in variables_input if v not in numericas_input]


# Tratamiento de valores atipicos
resultados = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}

for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]
    
# Visualizamos relacion entre ausentes y missing
patron_perdidos(datos_input)

datos_input[variables_input].isna().sum()
prop_missingsVars = datos_input.isna().sum()/len(datos_input)

#Media de valores perdidos para cada una de las filas
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)
datos_input['prop_missings'].describe()
len(datos_input['prop_missings'].unique())

# Transformamos a categorica
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')

# Imputamos valores perdidos
for x in numericas_input:
    datos_input[x] = ImputacionCuant(datos_input[x], 'aleatorio')
    
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'aleatorio')

# Revisamos que no queden datos missing
datos_input.isna().sum()

# Guardamos los datos depurados
datosEleccionesDep = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datosEleccionesDep.pickle', 'wb') as archivo:
    pickle.dump(datosEleccionesDep, archivo)


###############################################################################
################################ ANALISIS #####################################
###############################################################################
with open('datosEleccionesDep.pickle', 'rb') as f:
    datos = pickle.load(f)
   
varObjCont = datos['AbstentionPtge']
varObjBin = datos['Izquierda']
datos_input = datos.drop(['AbstentionPtge', 'Izquierda'], axis = 1) 
variables = list(datos_input.columns)  

# Graficamos la V de cramer
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)

# Guardamos los datos de la V de cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjCont)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjCont.name, 'Vcramer': v_cramer},
                             ignore_index=True)
    
for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjBin)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjBin.name, 'Vcramer': v_cramer},
                             ignore_index=True)

# Ver graficamente efecto de dos variables cualitativas sobre la binaria
#mosaico_targetbinaria(datos_input['CCAA'], varObjBin, 'CCAA')
#boxplot_targetbinaria(datos_input['TotalCensus'], varObjBin,'Objetivo', 'TotalCensus')
#mosaico_targetbinaria(datos_input['Densidad'], varObjBin, 'Densidad')

    
# Ver graficamente efecto de dos variables cualitativas sobre la continua
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
matriz_corr = pd.concat([varObjCont, datos_input[numericas]], axis = 1).corr(method = 'pearson')
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
plt.figure(figsize=(15, 10))
sns.set(font_scale=1)
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".1f", cbar=True, mask=mask)
plt.title("Matriz de correlación")
plt.show()

datos_input.drop(['Population', 'totalEmpresas', 'Age_under19_Ptge', 'Age_over65_pct'], inplace=True, axis=1)

todo_cont = pd.concat([datos_input, varObjCont], axis = 1)
todo_bin = pd.concat([datos_input, varObjBin], axis = 1)

###############################################################################
######################## REGRESION LINEAL #####################################
###############################################################################

x_train, x_test, y_train, y_test = train_test_split(todo_cont, varObjCont,
                                                    test_size = 0.2, random_state = 29112002)


var_cont = ['TotalCensus', 'WomanPopulationPtge', 'UnemployLess25_Ptge', 
            'Unemploy25_40_Ptge', 'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge',
            'IndustryUnemploymentPtge', 'ConstructionUnemploymentPtge', 'ServicesUnemploymentPtge',
            'PobChange_pct', 'PersonasInmueble', 'Explotaciones']
var_categ = ['CCAA', 'ActividadPpal', 'Densidad']


interacciones = var_cont #+ var_categ
interacciones_unicas = list(itertools.combinations(interacciones, 2))

modeloStepAIC = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloBackAIC = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloForwAIC = lm_forward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'AIC')
modeloStepBIC = lm_stepwise(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloBackBIC = lm_backward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')
modeloForwBIC = lm_forward(y_train, x_train, var_cont, var_categ, interacciones_unicas, 'BIC')


#modeloBackAIC['Modelo'].summary()
Rsq(modeloBackAIC['Modelo'], y_train, modeloBackAIC['X'])
x_test_BAIC = crear_data_modelo(x_test,modeloBackAIC['Variables']['cont'],
                                modeloBackAIC['Variables']['categ'], modeloBackAIC['Variables']['inter'])
Rsq(modeloBackAIC['Modelo'], y_test, x_test_BAIC)
len(modeloBackAIC['Modelo'].params)


#modeloBackBIC['Modelo'].summary()
Rsq(modeloBackBIC['Modelo'], y_train, modeloBackBIC['X'])
x_test_BBIC = crear_data_modelo(x_test,modeloBackBIC['Variables']['cont'],
                                modeloBackBIC['Variables']['categ'], modeloBackBIC['Variables']['inter'])
Rsq(modeloBackBIC['Modelo'], y_test, x_test_BBIC)
len(modeloBackBIC['Modelo'].params)


#modeloStepAIC['Modelo'].summary()
Rsq(modeloStepAIC['Modelo'], y_train, modeloStepAIC['X'])
x_test_SAIC = crear_data_modelo(x_test,modeloStepAIC['Variables']['cont'],
                                modeloStepAIC['Variables']['categ'], modeloStepAIC['Variables']['inter'])
Rsq(modeloStepAIC['Modelo'], y_test, x_test_SAIC)
len(modeloStepAIC['Modelo'].params)


#modeloStepBIC['Modelo'].summary()
Rsq(modeloStepBIC['Modelo'], y_train, modeloStepBIC['X'])
x_test_SBIC = crear_data_modelo(x_test,modeloStepBIC['Variables']['cont'],
                                modeloStepBIC['Variables']['categ'], modeloStepBIC['Variables']['inter'])
Rsq(modeloStepBIC['Modelo'], y_test, x_test_SBIC)
len(modeloStepBIC['Modelo'].params)


#modeloForwAIC['Modelo'].summary()
Rsq(modeloForwAIC['Modelo'], y_train, modeloForwAIC['X'])
x_test_FAIC = crear_data_modelo(x_test,modeloForwAIC['Variables']['cont'],
                                modeloForwAIC['Variables']['categ'], modeloForwAIC['Variables']['inter'])
Rsq(modeloForwAIC['Modelo'], y_test, x_test_FAIC)
len(modeloForwAIC['Modelo'].params)


#modeloForwBIC['Modelo'].summary()
Rsq(modeloForwBIC['Modelo'], y_train, modeloForwBIC['X'])
x_test_FBIC = crear_data_modelo(x_test,modeloForwBIC['Variables']['cont'],
                                modeloForwBIC['Variables']['categ'], modeloForwBIC['Variables']['inter'])
Rsq(modeloForwBIC['Modelo'], y_test, x_test_FBIC)
len(modeloForwBIC['Modelo'].params)

ganador = modeloBackBIC


######### SELECCION ALEATORIA
# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {'Formula': [],'Variables': []}

for x in range(30):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train, 
                                                            test_size = 0.3, random_state = 29112002 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train2.astype(int), x_train2, var_cont, var_categ, interacciones_unicas, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))
    
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))

# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(frec_ordenada['Formula'][1])]

# Salidas de los modelos que más se repiten
var_1
var_2
ganador['Variables']

# Selección del ganador
results = pd.DataFrame({'Rsquared': [], 'Resample': [], 'Modelo': []})
for rep in range(50):
    modelo1 = validacion_cruzada_lm(5, x_train, y_train, ganador['Variables']['cont']
              , ganador['Variables']['categ'], ganador['Variables']['inter'])
    modelo2 = validacion_cruzada_lm(5, x_train, y_train, var_1['cont'], var_1['categ'], var_1['inter'])
    modelo3 = validacion_cruzada_lm(5, x_train, y_train, var_2['cont'], var_2['categ'], var_2['inter'])
    
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 
        , 'Resample': ['Rep' + str((rep + 1))]*5*3
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 
    })
    results = pd.concat([results, results_rep], axis = 0)


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

# Construir el modelo si uno de los ganadores es el de aleatoria
modelo = lm(y_train, x_train, var_2['cont'], var_2['categ'], var_2['inter'])
modelo['Modelo'].summary()


###############################################################################
######################## REGRESION LOGISTICA ##################################
###############################################################################
# Comprobamos que izquierda solo toma valor 1 y 0
pd.DataFrame({'n': varObjBin.value_counts(), '%': varObjBin.value_counts(normalize = True)})


# Obtengo la particion
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(todo_bin, varObjBin, test_size = 0.2, random_state = 29112002)
# Indico que la variable respuesta es numérica (hay que introducirla en el algoritmo de phython tal y como la va a tratar)
y_train_log, y_test_log = y_train_log.astype(int), y_test_log.astype(int)

var_cont_log = ['TotalCensus', 'WomanPopulationPtge', 'UnemployLess25_Ptge', 
                'Unemploy25_40_Ptge', 'UnemployMore40_Ptge', 'AgricultureUnemploymentPtge', 'Explotaciones']
var_categ_log = ['CCAA', 'ActividadPpal', 'Densidad']

interacciones_log = var_cont_log #+ var_categ_log
interacciones_unicas_log = list(itertools.combinations(interacciones_log, 2))

modeloLogStepAIC = glm_stepwise(y_train_log, x_train_log, var_cont_log, var_categ_log, interacciones_unicas_log, 'AIC')
modeloLogStepBIC = glm_stepwise(y_train_log, x_train_log, var_cont_log, var_categ_log, interacciones_unicas_log, 'BIC')
modeloLogForwAIC = glm_forward(y_train_log, x_train_log, var_cont_log, var_categ_log, interacciones_unicas_log, 'AIC')
modeloLogForwBIC = glm_forward(y_train_log, x_train_log, var_cont_log, var_categ_log, interacciones_unicas_log, 'BIC')
modeloLogBackAIC = glm_backward(y_train_log, x_train_log, var_cont_log, var_categ_log, interacciones_unicas_log, 'AIC')
modeloLogBackBIC = glm_backward(y_train_log, x_train_log, var_cont_log, var_categ_log, interacciones_unicas_log, 'BIC')


#(modeloLogStepAIC['Modelo'], y_train_log, modeloLogStepAIC['X'])
pseudoR2(modeloLogStepAIC['Modelo'], modeloLogStepAIC['X'], y_train_log)
x_test_LSAIC = crear_data_modelo(x_test_log,modeloLogStepAIC['Variables']['cont'],
                                modeloLogStepAIC['Variables']['categ'], modeloLogStepAIC['Variables']['inter'])
pseudoR2(modeloLogStepAIC['Modelo'], x_test_LSAIC, y_test_log)
len(modeloLogStepAIC['Modelo'].coef_[0])


#summary_glm(modeloLogBackAIC['Modelo'], y_train_log, modeloLogBackAIC['X'])
pseudoR2(modeloLogBackAIC['Modelo'], modeloLogBackAIC['X'], y_train_log)
x_test_LBAIC = crear_data_modelo(x_test_log,modeloLogBackAIC['Variables']['cont'],
                                modeloLogBackAIC['Variables']['categ'], modeloLogBackAIC['Variables']['inter'])
pseudoR2(modeloLogBackAIC['Modelo'], x_test_LBAIC, y_test_log)
len(modeloLogBackAIC['Modelo'].coef_[0])


#summary_glm(modeloLogForwAIC['Modelo'], y_train_log, modeloLogForwAIC['X'])
pseudoR2(modeloLogForwAIC['Modelo'], modeloLogForwAIC['X'], y_train_log)
x_test_LFAIC = crear_data_modelo(x_test_log,modeloLogForwAIC['Variables']['cont'],
                                modeloLogForwAIC['Variables']['categ'], modeloLogForwAIC['Variables']['inter'])
pseudoR2(modeloLogForwAIC['Modelo'], x_test_LFAIC, y_test_log)
len(modeloLogForwAIC['Modelo'].coef_[0])


#summary_glm(modeloLogStepBIC['Modelo'], y_train_log, modeloLogStepBIC['X'])
pseudoR2(modeloLogStepBIC['Modelo'], modeloLogStepBIC['X'], y_train_log)
x_test_LSBIC = crear_data_modelo(x_test_log,modeloLogStepBIC['Variables']['cont'],
                                modeloLogStepBIC['Variables']['categ'], modeloLogStepBIC['Variables']['inter'])
pseudoR2(modeloLogStepBIC['Modelo'], x_test_LSBIC, y_test_log)
len(modeloLogStepBIC['Modelo'].coef_[0])


#summary_glm(modeloLogBackBIC['Modelo'], y_train_log, modeloLogBackBIC['X'])
pseudoR2(modeloLogBackBIC['Modelo'], modeloLogBackBIC['X'], y_train_log)
x_test_LBBIC = crear_data_modelo(x_test_log,modeloLogBackBIC['Variables']['cont'],
                                modeloLogBackBIC['Variables']['categ'], modeloLogBackBIC['Variables']['inter'])
pseudoR2(modeloLogBackBIC['Modelo'], x_test_LBBIC, y_test_log)
len(modeloLogBackBIC['Modelo'].coef_[0])


#summary_glm(modeloLogForwBIC['Modelo'], y_train_log, modeloLogForwBIC['X'])
pseudoR2(modeloLogForwBIC['Modelo'], modeloLogForwBIC['X'], y_train_log)
x_test_LFBIC = crear_data_modelo(x_test_log,modeloLogForwBIC['Variables']['cont'],
                                modeloLogForwBIC['Variables']['categ'], modeloLogForwBIC['Variables']['inter'])
pseudoR2(modeloLogForwBIC['Modelo'], x_test_LFBIC, y_test_log)
len(modeloLogForwBIC['Modelo'].coef_[0])


ganador_log = modeloLogStepAIC

######### SELECCION ALEATORIA
# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas_log = {'Formula': [],'Variables': []}

for x in range(10):
    print('---------------------------- iter: ' + str(x))
    
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train_log2, x_test_log2, y_train_log2, y_test_log2 = train_test_split(x_train_log, y_train_log, 
                                                            test_size = 0.3, random_state = 29112002 + x)
    
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train_log2.astype(int), x_train_log2, var_cont_log, var_categ_log, interacciones_unicas_log, 'BIC')
    
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas_log['Variables'].append(modelo['Variables'])
    variables_seleccionadas_log['Formula'].append(sorted(modelo['Modelo'].model.exog_names))
    
variables_seleccionadas_log['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas_log['Formula']))

# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias_log = Counter(variables_seleccionadas_log['Formula'])
frec_ordenada_log = pd.DataFrame(list(frecuencias_log.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada_log = frec_ordenada_log.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
var_1_log = variables_seleccionadas_log['Variables'][variables_seleccionadas_log['Formula'].index(frec_ordenada_log['Formula'][0])]
var_2_log = variables_seleccionadas_log['Variables'][variables_seleccionadas_log['Formula'].index(frec_ordenada_log['Formula'][1])]

# Salidas de los modelos que más se repiten
var_1_log
var_2_log
ganador_log['Variables']


results_log = pd.DataFrame({'Rsquared': [], 'Resample': [], 'Modelo': []})
for rep in range(20):
    modelo1 = validacion_cruzada_lm(5, x_train_log, y_train_log, ganador_log['Variables']['cont']
              , ganador_log['Variables']['categ'], ganador_log['Variables']['inter'])
    modelo2 = validacion_cruzada_lm(5, x_train_log, y_train_log, var_1['cont'], var_1['categ'], var_1['inter'])
    modelo3 = validacion_cruzada_lm(5, x_train_log, y_train_log, var_2['cont'], var_2['categ'], var_2['inter'])
    
    results_rep = pd.DataFrame({
        'Rsquared': modelo1 + modelo2 + modelo3 
        , 'Resample': ['Rep' + str((rep + 1))]*5*3
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 
    })
    results = pd.concat([results, results_rep], axis = 0)
    
#### Determinamos el ganador
# Hacemos el boxplot
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


# Calculamos el ROC


# Calculamos el AUC o area bajo la curva