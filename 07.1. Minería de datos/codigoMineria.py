import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
os.chdir("C:\\Users\\Hugo\\OneDrive\\Desktop\\Entregas\\EntregasMaster\\07.1. Minería de datos")
from FuncionesMineria import *


# Leemos los datos, los mostramos y mostramos sus tipos
datos = pd.read_excel("DatosElecciones.xlsx", sheet_name='DatosEleccionesEspaña')
print(datos.head(5))
print(datos.dtypes)

# Dividimos las variables en categóricas y numéricas, y comprobamos que tengan el tipo correcto.
variables = list(datos.columns)
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
for v in categoricas:
    datos[v] = datos[v].replace('nan', np.nan)

# Categoricas
# Agrupamos construccion e industria en otro
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Construccion': 'Otro', 'Industria': 'Otro'})

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
print(datos_input['TotalCensus'].unique())

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

