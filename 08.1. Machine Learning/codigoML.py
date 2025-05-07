import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

SEMILLA = 123456

# Importamos los datos
df = pd.read_excel('datos_tarea25.xlsx')
# Eliminamos los duplicados que nos encontramos
df = df.drop_duplicates()

df.dtypes
print(df.describe())
df.nunique()

# Analizamos missing
df.isnull().sum()

# Mapeos binarios
df['Leather interior'] = df['Leather interior'].map({'Yes': 1, 'No': 0})
df['Right wheel'] = df['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1})
df['Automatic'] = df['Gear box type'].map({'Automatic': 1, 'Tiptronic': 0})
df.drop(['Wheel', 'Gear box type'], axis=1, inplace=True)

# Color binario
df['Color'] = df['Color'].map({'White': 1, 'Black': 0}).astype(int)

# Separar turbo y volumen de motor
df['Turbo'] = df['Engine volume'].str.contains('Turbo', case=False, na=False).astype(int)
df['Engine volume'] = df['Engine volume'].str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)

# Limpiar Levy y Mileage
df['Levy'] = df['Levy'].apply(lambda x: 0 if x=='-' else x).astype(int)
#df.drop('Levy', axis=1, inplace=True)
df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False).str.replace(',', '').astype(int)

# Definir variables
nums = ['Price', 'Prod. year', 'Leather interior', 'Engine volume', 'Mileage',
        'Airbags', 'Cylinders', 'Right wheel', 'Automatic', 'Turbo', 'Levy']
cats = ['Manufacturer', 'Category', 'Fuel type', 'Drive wheels']
target = "Color"

# Analizamos distribución de categóricas para ver si es necesario reclasificar
for col in cats:
    print(df[col].value_counts(normalize=True).round(3) * 100)
    
# Reclasificamos 4x4 y Rear como una única categoría.
# Al tener ya solo dos, convertimos a una variable numérica binaria
df['Front_drive'] = (df['Drive wheels'] == 'Front').astype(int)
df.drop('Drive wheels', axis=1, inplace=True)
cats.remove('Drive wheels')
nums.append('Front_drive')

# Buscamos si hubiera outliers
descriptivos = df.describe()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(descriptivos)

# Vemos cuantos valores tienen engine volume a 0
(df['Engine volume']==0).sum()
df = df[df['Engine volume'] != 0]

# Detectamos outliers
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outliers_levy = detect_outliers(df, 'Levy')
outliers_mileage = detect_outliers(df, 'Mileage')
print(f"Outliers en Levy: {len(outliers_levy)}. Outliers en mileage: {len(outliers_mileage)}")

df1 = df[~df.isin(outliers_levy)].dropna()
df = df1[~df1.isin(outliers_mileage)].dropna()

# Crear dummies para las categóricas
df_dummies = pd.get_dummies(df, columns=cats, drop_first=True)
df_dummies.dropna(inplace=True)

corr_matrix = df[nums].corr()
plt.figure(figsize=(12, 8))  # Tamaño del gráfico
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# Eliminamos Cylinders por alta correlacion
df_dummies.drop(columns=['Cylinders'], inplace=True)
nums.remove('Cylinders')

# Separar X, y
X = df_dummies.drop(columns=[target])
Y = df_dummies[target]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=SEMILLA)

# Estandarizar después de dividir
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[nums] = scaler.fit_transform(X_train[nums])
X_test_scaled[nums] = scaler.transform(X_test[nums])


# Vamos a ajustar el SVM. Definimos los gridsearch y los modelos y los ajustamos
param_grid_linear = {'C': [0.1, 0.5, 1, 2, 5, 10, 100, 500]}
param_grid_poly = {'C': [0.1, 0.5, 1, 2, 5, 10, 100, 500], 'degree': [2, 3, 4], 'coef0': [0, 1, 2]}
param_grid_rbf = {'C': [0.1, 0.5, 1, 2, 5, 10, 100, 500], 'gamma': [0.1, 0.5, 1, 2, 5]}

grid_linear = GridSearchCV(SVC(kernel='linear', probability=True), param_grid_linear, refit=True, cv=10, verbose=3)
grid_poly = GridSearchCV(SVC(kernel='poly', probability=True), param_grid_poly, refit=True, cv=10, verbose=3)
grid_rbf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid_rbf, refit=True, cv=10, verbose=3)

grid_linear.fit(X_train_scaled, y_train)
grid_poly.fit(X_train_scaled, y_train)
grid_rbf.fit(X_train_scaled, y_train)

# Mostramos los mejores parámetros y el modelo ajustado
print("Mejor parámetro C para kernel lineal:", grid_linear.best_params_)
print("Mejor parámetro C y otros para kernel polinomial:", grid_poly.best_params_)
print("Mejor parámetro C y gamma para kernel RBF:", grid_rbf.best_params_)

# Hacemos predicciones con el mejor modelo de cada kernel
linear_predictions = grid_linear.predict(X_test_scaled)
poly_predictions = grid_poly.predict(X_test_scaled)
rbf_predictions = grid_rbf.predict(X_test_scaled)

# Mostramos el reporte de clasificacion
print("\nReporte de clasificación con el modelo lineal:")
print(classification_report(y_test, linear_predictions))

print("\nReporte de clasificación con el modelo polinomial:")
print(classification_report(y_test, poly_predictions))

print("\nReporte de clasificación con el modelo RBF:")
print(classification_report(y_test, rbf_predictions))

# Matriz de confusion y accuracy
linear_cm = confusion_matrix(y_test, linear_predictions)
poly_cm = confusion_matrix(y_test, poly_predictions)
rbf_cm = confusion_matrix(y_test, rbf_predictions)

linear_accuracy = (linear_cm[0, 0] + linear_cm[1, 1]) / linear_cm.sum()
poly_accuracy = (poly_cm[0, 0] + poly_cm[1, 1]) / poly_cm.sum()
rbf_accuracy = (rbf_cm[0, 0] + rbf_cm[1, 1]) / rbf_cm.sum()

print("Modelo lineal")
plt.figure(figsize=(8, 6))
linear_cm = confusion_matrix(y_test, linear_predictions)
sns.heatmap(linear_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Black (0)', 'White (1)'], yticklabels=['Black (0)', 'White (1)'], cbar=False)
plt.title("Matriz de Confusión - Modelo Lineal")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
print("Accuracy modelo lineal:", linear_accuracy)

print("Modelo polinomial")
plt.figure(figsize=(8, 6))
linear_cm = confusion_matrix(y_test, linear_predictions)
sns.heatmap(poly_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Black (0)', 'White (1)'], yticklabels=['Black (0)', 'White (1)'], cbar=False)
plt.title("Matriz de Confusión - Modelo polinomial")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
print("Accuracy modelo polinomial:", poly_accuracy)

print("Modelo RBF")
plt.figure(figsize=(8, 6))
linear_cm = confusion_matrix(y_test, linear_predictions)
sns.heatmap(rbf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Black (0)', 'White (1)'], yticklabels=['Black (0)', 'White (1)'], cbar=False)
plt.title("Matriz de Confusión - Modelo RBG")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()
print("Accuracy modelo RBF:", rbf_accuracy)


# 7. Representación gráfica de la búsqueda de parámetros para cada kernel

# LINEAL
aux_linear = pd.DataFrame(grid_linear.cv_results_)
plt.figure(figsize=(8, 6))
plt.scatter(aux_linear['param_C'], aux_linear['mean_test_score'], color='b', alpha=0.9)
plt.xlabel('Parametro C')
plt.ylabel('Accuracy')
plt.title('Precisión media SVM con kernel lineal en función del parámetro C')
plt.show()

# POLINOMIAL
aux_poly = pd.DataFrame(grid_poly.cv_results_)
sns.relplot(x="param_C", y="mean_test_score", hue="param_degree", size="param_coef0", data=aux_poly)
plt.title('Precisión media SVM con kernel polinomial')
plt.show()

# RBF
aux_rbf = pd.DataFrame(grid_rbf.cv_results_)
plt.scatter(aux_rbf['param_C'], aux_rbf['mean_test_score'], c=aux_rbf['param_gamma'], cmap='viridis')
plt.xlabel('Parametro C')
plt.ylabel('Accuracy')
plt.title('Precisión media SVM con kernel RBF en función del parámetro C')
plt.show()

# Predicciones de probabilidad para ROC AUC
y_prob_linear = grid_linear.predict_proba(X_test_scaled)[:, 1]
y_prob_poly = grid_poly.predict_proba(X_test_scaled)[:, 1]
y_prob_rbf = grid_rbf.predict_proba(X_test_scaled)[:, 1]

# Calcular ROC AUC
fpr_linear, tpr_linear, _ = roc_curve(y_test, y_prob_linear)
fpr_poly, tpr_poly, _ = roc_curve(y_test, y_prob_poly)
fpr_rbf, tpr_rbf, _ = roc_curve(y_test, y_prob_rbf)

roc_auc_linear = auc(fpr_linear, tpr_linear)
roc_auc_poly = auc(fpr_poly, tpr_poly)
roc_auc_rbf = auc(fpr_rbf, tpr_rbf)

# Graficar las curvas ROC
plt.figure(figsize=(10, 6))

plt.plot(fpr_linear, tpr_linear, color='blue', lw=2, label='Linear kernel (AUC = %0.2f)' % roc_auc_linear)
plt.plot(fpr_poly, tpr_poly, color='green', lw=2, label='Polynomial kernel (AUC = %0.2f)' % roc_auc_poly)
plt.plot(fpr_rbf, tpr_rbf, color='red', lw=2, label='RBF kernel (AUC = %0.2f)' % roc_auc_rbf)

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para los tres kernels SVM')
plt.legend(loc='lower right')
plt.show()


# Elegimos el mejor modelo
best_svm_model = grid_linear.best_estimator_

# Una vez elegido el mejor, le aplicamos un bagging
bagging_model = BaggingClassifier(estimator=best_svm_model, 
                                 n_estimators=50,
                                 max_samples=0.7,  
                                 max_features=1.0, 
                                 random_state=SEMILLA)

bagging_model.fit(X_train_scaled, y_train)
y_pred_bagging = bagging_model.predict(X_test_scaled)

# Evaluamos el rendimiento
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f'Precisión del modelo con Bagging: {accuracy_bagging}')

# Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred_bagging))

# Mostrar la matriz de confusión
cm_bagging = confusion_matrix(y_test, y_pred_bagging)
plt.figure(figsize=(8, 6))
linear_cm = confusion_matrix(y_test, linear_predictions)
sns.heatmap(cm_bagging, annot=True, fmt='d', cmap='Blues', xticklabels=['Black (0)', 'White (1)'], yticklabels=['Black (0)', 'White (1)'], cbar=False)
plt.title("Matriz de Confusión - Bagging")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()





###############################################
# Ahora haremos el stacking en profundidad ####
# Inicializamos los modelos base y definimos los clasificadores en una lista
model_1 = RandomForestClassifier(random_state=SEMILLA)
model_2 = LogisticRegression(random_state=SEMILLA, max_iter=1000)
model_3 = KNeighborsClassifier()

base_models = [('rf', model_1), ('lr', model_2), ('knn', model_3)]

# Hacemos un metaclasificador
stacking_model = StackingClassifier(
    estimators=base_models,  # Modelos base
    final_estimator=RandomForestClassifier(random_state=42),  
    cv=5  
)

# Entrenmos el modelo y hacemos predicciones
stacking_model.fit(X_train_scaled, y_train)
y_pred = stacking_model.predict(X_test_scaled)

# Mostramos el report y calculamos el rendimiento de cada clasificador y el global
print("Resultados del modelo de stacking:")
print(classification_report(y_test, y_pred))

# Evaluar el rendimiento de cada clasificador base por separado
for name, clf in stacking_model.named_estimators_.items():
    y_pred_base = clf.predict(X_test)
    print(f"\nResultados del Clasificador Base {name}:")
    print(classification_report(y_test, y_pred_base))




###############################################
# Comparamos los modelos del SVM, el bagging y el stacking
# Obtener las predicciones para SVM
y_pred_svm = best_svm_model.predict(X_test_scaled)
report_svm = classification_report(y_test, y_pred_svm, output_dict=True)

# Obtener las predicciones para Bagging
y_pred_bagging = bagging_model.predict(X_test_scaled)
report_bagging = classification_report(y_test, y_pred_bagging, output_dict=True)

# Obtener las predicciones para Stacking
y_pred_stacking = stacking_model.predict(X_test_scaled)
report_stacking = classification_report(y_test, y_pred_stacking, output_dict=True)

# Crear la tabla comparativa
kpis = []

# Modelo SVM
kpis.append({
    'Model': 'SVM',
    'Accuracy': report_svm['accuracy'],
    'Precision (0.0)': report_svm['0.0']['precision'],
    'Recall (0.0)': report_svm['0.0']['recall'],
    'F1-Score (0.0)': report_svm['0.0']['f1-score'],
    'Precision (1.0)': report_svm['1.0']['precision'],
    'Recall (1.0)': report_svm['1.0']['recall'],
    'F1-Score (1.0)': report_svm['1.0']['f1-score']
})

# Modelo Bagging
kpis.append({
    'Model': 'Bagging',
    'Accuracy': report_bagging['accuracy'],
    'Precision (0.0)': report_bagging['0.0']['precision'],
    'Recall (0.0)': report_bagging['0.0']['recall'],
    'F1-Score (0.0)': report_bagging['0.0']['f1-score'],
    'Precision (1.0)': report_bagging['1.0']['precision'],
    'Recall (1.0)': report_bagging['1.0']['recall'],
    'F1-Score (1.0)': report_bagging['1.0']['f1-score']
})

# Modelo Stacking
kpis.append({
    'Model': 'Stacking',
    'Accuracy': report_stacking['accuracy'],
    'Precision (0.0)': report_stacking['0.0']['precision'],
    'Recall (0.0)': report_stacking['0.0']['recall'],
    'F1-Score (0.0)': report_stacking['0.0']['f1-score'],
    'Precision (1.0)': report_stacking['1.0']['precision'],
    'Recall (1.0)': report_stacking['1.0']['recall'],
    'F1-Score (1.0)': report_stacking['1.0']['f1-score']
})

# Convertir la lista de KPIs a un DataFrame
kpis_df = pd.DataFrame(kpis)

# Mostrar la tabla de comparación
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(kpis_df)
