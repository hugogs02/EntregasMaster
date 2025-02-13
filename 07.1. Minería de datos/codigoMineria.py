import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\Hugo\\OneDrive\\Desktop\\Entregas\\EntregasMaster\\07.1. Minería de datos")
from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali)


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

# Comprobamos si hay valores faltantes
print(datos[variables].isna().sum())

# Corregimos los errores
#######################################################################
###AQUI CORREGIMOS ERRORES
#######################################################################

varObjCont = datos['AbstentionPtge']
varObjBin = datos['Izquierda']
datos_input = datos.drop(['AbstentionPtge', 'Izquierda'], axis=1)

numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns
categoricas_input = [v for v in variables if v not in numericas_input]