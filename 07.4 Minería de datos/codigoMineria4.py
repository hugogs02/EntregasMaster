import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from optbinning import Scorecard, BinningProcess, OptimalBinning

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import geopandas as gpd
from libpysal import weights
from esda.moran import Moran

# Imnportamos el conjunto de daros "germacredit"
dt=pd.read_csv('germancredit.csv')

#Recodifico esta variable creditability (variable objetivo) para que sea binaria
dt["y"]=0
dt.loc[dt["creditability"]=="good",["y"]]=0
dt.loc[dt["creditability"]=="bad", ["y"]]=1
dt.drop(labels='creditability',inplace=True, axis=1)

# Creo la muestra de entrenamiento y de test
dt_train, dt_test = train_test_split(dt, stratify= dt["y"], test_size=.25, random_state=1234)

variable="age.in.years"
X=dt_train[variable].values
Y=dt_train['y'].values
optb = OptimalBinning(name=variable, dtype="numerical", solver="cp")
optb.fit(X, Y)
optb.splits
binning_table = optb.binning_table
binning_table.build()

##########################
### PREGUNTA 9 ###########
# Cargar los datos
data = pd.read_excel("DatosPractica_Scoring.xlsx")

# Explorar los datos
print(data.info())
print(data.describe())

# Feature Engineering
data['Debt_to_Income'] = data['Avgexp'] / (data['Income'] * 10_000)
data['Stability'] = data['Cur_add'] + data['Major'] + data['Active']

# Filtrar datos: separar aceptados, rechazados y nuevos clientes
accepted = data[data['Cardhldr'] == 1].copy()
rejected = data[data['Cardhldr'] == 0].copy()
new_clients = data[data['Cardhldr'].isna()].copy()

# Verificar valores nulos
accepted.dropna(subset=['default'], inplace=True)

# Selección de variables predictoras
features = ['Age', 'Income', 'Exp_Inc', 'Avgexp', 'Ownrent', 'Selfempl', 'Depndt', 'Inc_per', 'Cur_add', 'Major', 'Active', 'Debt_to_Income', 'Stability']
X = accepted[features]
y = accepted['default']

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Optimización de hiperparámetros
param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
gbc = GradientBoostingClassifier()
grid_search = GridSearchCV(gbc, param_grid, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluar el modelo
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))
print(classification_report(y_test, y_pred))

# Predecir para nuevos clientes
new_X = new_clients[features]
new_X_scaled = scaler.transform(new_X)
new_clients['default_pred'] = best_model.predict(new_X_scaled)

# Mostrar los clientes aprobados
approved_clients = new_clients[new_clients['default_pred'] == 0]
print("Clientes aprobados:", approved_clients['ID'].tolist())



##################################
####### EJERCICIO 14 #############
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal import weights
import esda
import spreg as spreg

gdfm = gpd.read_file("Munic04_ESP.shp")

wq = weights.contiguity.Queen.from_dataframe(gdfm)
wq.transform = "R"
# MODELO (A)

modelo_A = spreg.OLS(
    # Dependent variable
    gdfm[["TASA_PARO"]].values,
    # Independent variables
    gdfm[["RENTPCAP07"]].values,
    # Dependent variable name
    name_y="TASA_PARO",
    # Independent variable name
    name_x=["RENTA_PERCAPITA"])

gdfm["residual"] = modelo_A.u

moran = esda.moran.Moran(gdfm["residual"], wq)
print("I de moran:", moran.I.round(3))
print("p-valor:", moran.p_sim)


# MODELO (B)

modelo_B = spreg.GM_Error_Het(
    # Dependent variable
    gdfm[["TASA_PARO"]].values,
    # Independent variables
    gdfm[["RENTPCAP07"]].values,
    # Spatial weights matrix
    w=wq,
    # Dependent variable name
    name_y="TASA_PARO",
    # Independent variable name
    name_x=["RENTA_PERCAPITA"])  
  
gdfm["mLagresidual"] = modelo_B.e_filtered

moran = esda.moran.Moran(gdfm["mLagresidual"], wq)
print("I de moran:", moran.I.round(3))
print("p-valor:", moran.p_sim)



##################################
####### EJERCICIO 15 #############
df = pd.read_csv('Data_Housing_Madrid.csv')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

# Estadísticas varias
num_total_viviendas = len(gdf)
gdf_historical = gdf[gdf['historical'] == 1].copy()
num_historical_viviendas = len(gdf_historical)
precio_mediano = gdf_historical['house.price'].median()
precio_maximo = gdf_historical['house.price'].max()

# Matriz de pesos
w_hy = weights.distance.DistanceBand.from_dataframe(
    gdf_historical, threshold=0.00225, alpha=-1, binary=False)

# Viviendas sin vecinas
sin_vecinas = np.sum([len(neighbors) == 0 for neighbors in w_hy.neighbors.values()])

# Mediana de vecinas por vivienda
num_vecinas = [len(neighbors) for neighbors in w_hy.neighbors.values()]
mediana_vecinas = np.median(num_vecinas)

# Indice de Moran
y = gdf_historical['house.price'].values
moran = Moran(y, w_hy)

# Más datos relevantes
indice_moran = round(moran.I, 3)
pvalor_moran = round(moran.p_sim, 3)


if pvalor_moran < 0.05:
    conclusion = "analizar el precio de las viviendas vecinas Sí ayuda a estimar el precio de una vivienda en el centro histórico de Madrid"
else:
    conclusion = "no es posible determinar si el precio de las viviendas vecinas ayudan o no ayudan a estimar el precio de una vivienda en el centro histórico de Madrid"


print(f"a) Total: {num_total_viviendas}")
print(f"b) Casco histórico: {num_historical_viviendas}")
print(f"c) Precio mediano: {precio_mediano:.2f} €/m2")
print(f"d) Precio máximo: {precio_maximo:.2f} €/m2")
print(f"e) Sin vecinas: {sin_vecinas}")
print(f"f) Mediana de viviendas vecinas: {mediana_vecinas}")
print(f"g) Índice de Moran: {indice_moran}")
print(f"h) P-valor: {pvalor_moran}")
print(f"i) Conclusión: {conclusion}")