import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.cluster.hierarchy as sch
os.chdir('C:\\Users\\Hugo\\OneDrive\\Desktop\\Entregas\\EntregasMaster\\07.2 Minería de datos')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from scipy.spatial import distance
from FuncionesMineria2 import *

# Importamos el archivo, extraemos las variables
penguins = sns.load_dataset("penguins")
penguins.head()
variables = penguins.columns
variables_numericas = [x for x in variables if x not in ['species', 'island', 'sex']]
penguins_nums = penguins[variables_numericas]

# Obtenemos los descriptivos
descriptivos = penguins_nums.describe().T
for num in variables_numericas:
    descriptivos.loc[num, "Varianza"] = penguins_nums[num].var()
    descriptivos.loc[num, "Coef. Variación"] = (penguins_nums[num].std() / penguins_nums[num].mean() )
    descriptivos.loc[num, "Missing"] = (penguins_nums[num].isna().sum())

# Eliminamos las filas con variablers NAN (solo hay 11, no perdemos cantidad significativa de datos)
penguins.dropna(inplace=True)
penguins.isna().sum()

# Mapeamos el género
penguins['sex'] = penguins['sex'].map({'Male': 1, 'Female': 0})

# Quitamos la especie y la isla
penguins_original = penguins.copy()
penguins.drop(['species', 'island'], axis=1, inplace=True)
variables = list(penguins.columns)

# Dibujamos la matriz de correlación
corr = penguins.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

###################################
### ANALISIS PCA
# Estandarizacion de los datos
penguins_std = pd.DataFrame(StandardScaler().fit_transform(penguins),
    columns=['{}_z'.format(variable) for variable in variables], index=penguins.index )
penguins_std.head()

#Creamos el PCA y lo aplicamos
pca1 = PCA(n_components=len(variables))
fit1 = pca1.fit(penguins_std)
autovalores = fit1.explained_variance_ # Obtenemos autovalores
autovalores

var_explicada = fit1.explained_variance_ratio_ #Obtenemos varianza explicada
var_explicada

var_acumulada = np.cumsum(var_explicada) #Varianza explicada según agregamos componentes
var_acumulada

# Creamos dataframe y representamos
data = {'Autovalores': autovalores, 'Var.Explicada': var_explicada, 'Var. Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit1.n_components_+1)])
print(tabla)

plot_varianza_explicada(var_explicada, fit1.n_components_)

#Repetimos el PCA con los dos componentes elegidos
pca = PCA(n_components=2)
fit = pca.fit(penguins_std)
print(pca.components_)

autovectores = pd.DataFrame(pca.components_.T,
                            columns = ['Autovector {}'.format(i) for i in range(1, fit.n_components_+1)],
                            index = ['{}_z'.format(variable) for variable in variables])
print(autovectores)

# Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(penguins_std),
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=penguins_std.index)

# Añadimos las componentes principales a la base de datos estandarizada.
penguins_z_cp = pd.concat([penguins_std, resultados_pca], axis=1)
print(penguins_z_cp)

# Calculamos correlacion de las variables originales con las componentes elegidas
variables_cp = penguins_z_cp.columns

# Calculamos las correlaciones y seleccionamos las que nos interesan (variables contra componentes).
correlacion = pd.DataFrame(np.corrcoef(penguins_std.T, resultados_pca.T),
                           index = variables_cp, columns = variables_cp)

n_variables = fit.n_features_in_
correlaciones_penguins_con_cp = correlacion.iloc[:fit.n_features_in_, fit.n_features_in_:]
print(correlaciones_penguins_con_cp)

cos2 = correlaciones_penguins_con_cp**2
print(cos2)

# Dibujamos los diferentes graficos
plot_cos2_heatmap(cos2)
plot_corr_cos(fit.n_components, correlaciones_penguins_con_cp)
plot_cos2_bars(cos2)
plot_pca_scatter(pca, penguins_std, fit.n_components)


##############################################################################
############### CLUSTERING #############
############### JERARQUICO #############
########################################

# Creamos un mapa de calor
sns.clustermap(penguins, cmap='coolwarm', annot=True)
plt.show()

# Calculamos la matriz de distancias y la representamos visualmente
distance_matrix = distance.cdist(penguins_std, penguins_std, 'euclidean')
print("Distance Matrix:")
distance_small = distance_matrix[:5, :5]
#Index are added to the distance matrix
distance_small = pd.DataFrame(distance_small, index=penguins_std.index[:5], columns=penguins_std.index[:5])

distance_small_rounded = distance_small.round(2)
print(distance_small_rounded)
penguins_std[:2]

plt.figure(figsize=(8, 6))
penguins_distance = pd.DataFrame(distance_matrix, index = penguins_std.index, columns = penguins_std.index)
sns.heatmap(penguins_distance, annot=False, cmap="YlGnBu", fmt=".1f")
plt.show()

# Obtenemos la matriz de enlace ??¿?¿?¿
linkage = sns.clustermap(penguins_distance, cmap="YlGnBu", fmt=".1f", annot=False, method='average').dendrogram_row.linkage

order = pd.DataFrame(linkage, columns=['cluster_1', 'cluster_2', 'distance', 'new_count']).index
reordered_data = penguins_std.reindex(index=order, columns=order)
sns.heatmap(reordered_data, cmap="YlGnBu", fmt=".1f", cbar=False)
plt.show()    

# Obtenemos el dendograma
linkage_matrix = sch.linkage(penguins_std, method='ward')
dendrogram = sch.dendrogram(linkage_matrix, labels=penguins.index, leaf_font_size=9, leaf_rotation=90)
plt.show()

## Se eligen los clusteres: 4

# Asignamos cada observación a uno de los clusteres
num_clusters = 4
cluster_assignments = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')
print("Cluster Assignments:", cluster_assignments)
plt.show()

penguins['Cluster4'] = cluster_assignments
print(penguins["Cluster4"])


# Validamos el agrupamiento
pca = PCA(n_components=2)
principal_components = pca.fit_transform(penguins)
penguins_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
plt.figure(figsize=(10, 6))

# Loop through unique cluster assignments and plot data points with the same color
for cluster in np.unique(cluster_assignments):
    plt.scatter(penguins_pca.loc[cluster_assignments == cluster, 'PC1'],
                penguins_pca.loc[cluster_assignments == cluster, 'PC2'],
                label=f'Cluster {cluster}')
# Add labels to data points
for i, row in penguins_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(penguins.index[i]), fontsize=8)

plt.title("Gráfico de PCA 2D con asignaciones de cluster")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid()
plt.show()

############### CLUSTERING ####################################################
############## NO JERARQUICO #############
##########################################
# Elegimos 4 clusters
k=4 
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(penguins_std)
kmeans_cluster_labels = kmeans.labels_
print(kmeans_cluster_labels)

# Creamos el gráfico
plt.figure(figsize=(10, 6))

for cluster in np.unique(kmeans_cluster_labels):
    plt.scatter(penguins_pca.loc[kmeans_cluster_labels == cluster, 'PC1'],
                penguins_pca.loc[kmeans_cluster_labels == cluster, 'PC2'],
                label=f'Cluster {cluster}')

for i, row in penguins_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(penguins.index[i]), fontsize=8)

plt.title("Gráfico de PCA 2D con asignaciones K-means")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid()
plt.show()

############### VALIDACION ####################################################
##########################################
# Creamos y aplicamos el DBSCAN para buscar un k optimo

# Inicializamos eps y min_samples, y las variables de numeros de clusters y parametros
eps_values = np.linspace(0.1, 1.0, 10)
min_samples_values = np.arange(2, 12)
num_clusters_prev = None
eps_best = None
min_samples_best = None
num_iterations = 0
array_num_clusters = []

# Iteramos sobre los diferentes valores
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(penguins_std)
        num_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        
        array_num_clusters.append(num_clusters)
        if num_clusters != num_clusters_prev:
            print(f"Iteración {num_iterations + 1}: eps={eps}, min_samples={min_samples}, Número de clusters={num_clusters}")
            num_clusters_prev = num_clusters
            eps_best = eps
            min_samples_best = min_samples

        num_iterations += 1
        
max(set(array_num_clusters), key=array_num_clusters.count)

# Vamos a probar con 5 clusters
dbscan = DBSCAN(eps=0.9, min_samples=2)
dbscan.fit(penguins_std)
dbscan_labels = dbscan.labels_
print(dbscan_labels)

plt.figure(figsize=(10, 6))
for cluster in np.unique(dbscan_labels):
    plt.scatter(penguins_pca.loc[dbscan_labels == cluster, 'PC1'],
                penguins_pca.loc[dbscan_labels == cluster, 'PC2'],
                label=f'Cluster {cluster}')
for i, row in penguins_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(penguins.index[i]), fontsize=8)

plt.title("Gráfico de PCA 2D con asignaciones DBSCAN")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend()
plt.grid()
plt.show()

# Vamos a aplicar el metodo del codo
wcss = []
for k in range(1, 11):  
    kmeans = KMeans(n_clusters=k, random_state=123456)
    kmeans.fit(penguins_std)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Método del codo')
plt.xlabel('Número de cluster (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Aplicamos el método de las siluetas  ############ REVISAR
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(penguins_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(penguins_std, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 10), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Método de la silueta')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Vemos los valores de la silueta
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(penguins_std)
labels = kmeans.labels_

silhouette_values = silhouette_samples(penguins_std, labels)
silhouette_values

# Dibujamos el grafico
plt.figure(figsize=(8, 6))

y_lower = 10
for i in range(4):
    ith_cluster_silhouette_values = silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.get_cmap("Spectral")(float(i) / 4)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.title("Gráfico de Sileuta para los Clusters")
plt.xlabel("Valores del coeficiente de silueta")
plt.ylabel("Etiqueta de cluster")
plt.grid(True)
plt.show()

# Caracterizamos los clusters
penguins_std['label'] = labels
penguins_std_sort = penguins_std.sort_values(by="label")
penguins_std = penguins_std.set_index(labels)
penguins_std_sort['label']

# Calculamos los centroides
cluster_centroids = penguins_std_sort.groupby('label').mean()
cluster_centroids.round(2)


# Repetimos pero con los datos originales, sin estandarizar
penguins['label'] = labels
penguins_sort = penguins.sort_values(by="label")
penguins = penguins.set_index(labels)
penguins_sort['label']

cluster_centroids = penguins_sort.groupby('label').mean()
cluster_centroids.round(2)

penguins_original['cluster'] = labels
print(penguins_original.groupby(['cluster', 'species']).size())
print(penguins_original.groupby(['cluster', 'island']).size())
