import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

SEMILLA = 123456

# Importamos los datos
df = pd.read_excel('datos_tarea25.xlsx')
print("Shape inicial:", df.shape)
print("Duplicados:", df.duplicated().sum())
df = df.drop_duplicates()

print(df.describe(include='all'))
print("Nulos por columna (antes):\n", df.isna().sum())

# Mapeos binarios
df['Leather interior'] = df['Leather interior'].map({'Yes': 1, 'No': 0}).astype(int)
df['Right wheel'] = df['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1}).astype(int)
df['Automatic'] = df['Gear box type'].map({'Automatic': 1, 'Tiptronic': 0}).astype(int)
df.drop(['Wheel', 'Gear box type'], axis=1, inplace=True)

# Target binaria
df['Color'] = df['Color'].map({'White': 1, 'Black': 0}).astype(int)
target = "Color"

# Separar Turbo y volumen de motor
df['Turbo'] = df['Engine volume'].str.contains('Turbo', case=False, na=False).astype(int)
df['Engine volume'] = df['Engine volume'].str.extract(r'(\d+(?:\.\d+)?)')[0].astype(float)

# Limpiar Levy y Mileage
df['Levy'] = df['Levy'].apply(lambda x: 0 if x == '-' else x).astype(int)
df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False).str.replace(',', '').astype(int)

# Definir listas de variables (como en tu código)
nums = ['Price', 'Prod. year', 'Leather interior', 'Engine volume', 'Mileage',
        'Airbags', 'Cylinders', 'Right wheel', 'Automatic', 'Turbo', 'Levy']
cats = ['Manufacturer', 'Category', 'Fuel type', 'Drive wheels']

# Distribución categóricas (diagnóstico)
for col in cats:
    print(f"\nDistribución {col} (%):")
    print((df[col].value_counts(normalize=True).round(3) * 100).head(20))

# Reclasificamos tracción a binaria (como tenías)
df['Front_drive'] = (df['Drive wheels'] == 'Front').astype(int)
df.drop('Drive wheels', axis=1, inplace=True)
cats.remove('Drive wheels')
nums.append('Front_drive')

# Outliers básicos 
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Prod. year'])
plt.title('Boxplot de Prod. year'); plt.grid(True); plt.show()

df = df[df['Engine volume'] != 0]
df = df[~df['Prod. year'].isin([1943, 1986])]

print("Distribución de la variable objetivo (Color):")
print(df[target].value_counts(normalize=True))

def detect_outliers(_df, column):
    Q1 = _df[column].quantile(0.25)
    Q3 = _df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return _df[( _df[column] < lower_bound) | (_df[column] > upper_bound)]

outliers_levy = detect_outliers(df, 'Levy')
outliers_mileage = detect_outliers(df, 'Mileage')
print(f"Outliers en Levy: {len(outliers_levy)} | Outliers en Mileage: {len(outliers_mileage)}")
df = df.drop(index=outliers_levy.index.union(outliers_mileage.index))

# IMPUTACIÓN EXPLÍCITA
assert df[target].isna().sum() == 0, "La variable objetivo contiene nulos."

for c in cats:
    df[c] = df[c].fillna("Missing")
for c in nums:
    df[c] = df[c].fillna(df[c].median())

print("Nulos tras imputación (deberían ser 0):\n", df[nums + cats + [target]].isna().sum().sort_values(ascending=False).head(10))

# EDA BIDIMENSIONAL vs TARGET
# 1) Numéricas vs target (boxplots)
for col in nums:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[target], y=df[col])
    plt.title(f"{col} vs {target}")
    plt.xlabel(target); plt.ylabel(col)
    plt.tight_layout(); plt.show()

# 2) Numéricas vs target (densidades) – opcional
for col in nums:
    plt.figure(figsize=(6,3))
    sns.kdeplot(data=df, x=col, hue=target, common_norm=False)
    plt.title(f"Densidades de {col} por clase de {target}")
    plt.tight_layout(); plt.show()

# 3) Categóricas vs target: tasa (barra) + soporte (línea)
def rate_plot(_df, col):
    tmp = (_df.groupby(col)[target]
           .agg(rate="mean", n="size")
           .sort_values("n", ascending=False).reset_index())
    fig, ax1 = plt.subplots(figsize=(8,3))
    sns.barplot(data=tmp, x=col, y="rate", ax=ax1)
    ax1.set_ylabel(f"Tasa {target}=1"); ax1.set_xlabel(col)
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(ax1.get_xticks(), tmp["n"], marker="o")
    ax2.set_ylabel("n (soporte)")
    plt.title(f"Tasa de {target} por {col} (con soporte)")
    plt.tight_layout(); plt.show()

for col in cats:
    rate_plot(df, col)

# Dummies, correlación y split
df_dummies = pd.get_dummies(df, columns=cats, drop_first=True)

corr_matrix = df[nums].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación'); plt.show()

# Eliminamos Cylinders por alta correlación (tu criterio)
if 'Cylinders' in df_dummies.columns:
    df_dummies.drop(columns=['Cylinders'], inplace=True)
if 'Cylinders' in nums:
    nums.remove('Cylinders')

# Separar X, y
X = df_dummies.drop(columns=[target])
Y = df_dummies[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=SEMILLA, stratify=Y)

# Escalado SOLO en columnas numéricas (como tenías)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[nums] = scaler.fit_transform(X_train[nums])
X_test_scaled[nums] = scaler.transform(X_test[nums])


###############################################################################
# PUNTO 2: Ajuste de SVM con búsqueda paramétrica y comparación de kernels
###############################################################################
# Grids de búsqueda "gruesa"
param_grid_linear = {'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 200, 500, 1000]}
param_grid_poly = {
    'C': [0.01, 1, 10], #[0.01, 0.1, 1, 10, 100]
    'degree': [2, 3, 4, 5],
    'coef0': [0, 1, 2, 3]
}
param_grid_rbf = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

def run_gridsearch(name, estimator, param_grid, n_folds):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    grid = GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring={'acc':'accuracy','auc':'roc_auc'},
        refit='auc',
        cv=cv, n_jobs=-1, verbose=2
    )
    grid.fit(X_train_scaled, y_train)
    print(f"\n{name} – mejor combinación:", grid.best_params_)
    print(f"{name} – mejor AUC CV: {grid.best_score_:.4f}")
    return grid

grid_linear = run_gridsearch("Lineal", SVC(kernel='linear', probability=True), param_grid_linear, 5)
grid_poly   = run_gridsearch("Polinomial", SVC(kernel='poly', probability=True), param_grid_poly, 5)
grid_rbf    = run_gridsearch("RBF", SVC(kernel='rbf', probability=True), param_grid_rbf, 5)

# ================
# Búsqueda fina alrededor de los mejores
# ================
def refine_gridsearch(name, base_grid, fine_grid):
    print(f"\n{name} – refinando alrededor de {base_grid.best_params_}")
    grid = GridSearchCV(
        base_grid.estimator,
        param_grid=fine_grid,
        scoring={'acc':'accuracy','auc':'roc_auc'},
        refit='auc',
        cv=5, n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    print(f"{name} – mejor combinación refinada:", grid.best_params_)
    print(f"{name} – mejor AUC CV refinada: {grid.best_score_:.4f}")
    return grid

# LINEAL: refino C cerca de 0.01
fine_grid_linear = {'C': [0.005, 0.0075, 0.01, 0.015, 0.02]}
grid_linear = refine_gridsearch("Lineal", grid_linear, fine_grid_linear)

# POLINOMIAL: dejo degree=2, coef0=3 fijos, refino C cerca de 0.01
fine_grid_poly = {'C': [0.005, 0.0075, 0.01, 0.015, 0.02], 'degree': [2], 'coef0': [3]}
grid_poly = refine_gridsearch("Polinomial", grid_poly, fine_grid_poly)

# RBF: refino C y gamma alrededor de 10 y 0.001
fine_grid_rbf = {'C': [5, 7.5, 10, 15, 20], 'gamma': [0.0005, 0.00075, 0.001, 0.0015, 0.002]}
grid_rbf = refine_gridsearch("RBF", grid_rbf, fine_grid_rbf)

# Visualizaciones
# LINEAL
aux_linear = pd.DataFrame(grid_linear.cv_results_)
plt.figure(figsize=(8,6))
plt.plot(aux_linear['param_C'], aux_linear['mean_test_acc'], 'o-', label="Accuracy")
plt.plot(aux_linear['param_C'], aux_linear['mean_test_auc'], 's-', label="AUC")
plt.xscale("log"); plt.xlabel('C (log)')
plt.ylabel('Score medio CV'); plt.title('SVM Lineal – Accuracy y AUC vs C')
plt.legend(); plt.show()

# POLINOMIAL
aux_poly = pd.DataFrame(grid_poly.cv_results_)
plt.figure(figsize=(8,6))
sns.scatterplot(data=aux_poly, x="param_C", y="mean_test_auc",
                hue="param_degree", size="param_coef0", palette="viridis")
plt.xscale("log"); plt.xlabel('C (log)')
plt.ylabel('AUC medio CV')
plt.title('SVM Polinomial – AUC vs C, degree, coef0')
plt.show()

# RBF
aux_rbf = pd.DataFrame(grid_rbf.cv_results_)
pivot_auc = aux_rbf.pivot(index='param_gamma', columns='param_C', values='mean_test_auc')
plt.figure(figsize=(8,6))
sns.heatmap(pivot_auc, cmap="viridis", annot=True, fmt=".3f")
plt.title("SVM RBF – AUC medio CV"); plt.ylabel("gamma"); plt.xlabel("C")
plt.show()

# Comparación final
results_dict = {
    "Lineal": {
        "params": grid_linear.best_params_,
        "auc": grid_linear.best_score_,
        "acc": grid_linear.cv_results_['mean_test_acc'][grid_linear.best_index_]
    },
    "Polinomial": {
        "params": grid_poly.best_params_,
        "auc": grid_poly.best_score_,
        "acc": grid_poly.cv_results_['mean_test_acc'][grid_poly.best_index_]
    },
    "RBF": {
        "params": grid_rbf.best_params_,
        "auc": grid_rbf.best_score_,
        "acc": grid_rbf.cv_results_['mean_test_acc'][grid_rbf.best_index_]
    }
}
print("\nResumen comparativo (validación cruzada):")
print(pd.DataFrame(results_dict).T)

# Selección final del mejor modelo
best_svm_model = max([grid_linear, grid_poly, grid_rbf], key=lambda g: g.best_score_).best_estimator_
print("\nMejor modelo final elegido por AUC en CV:", best_svm_model)

###############################################################################
# PUNTO 3: Evaluación baseline y Bagging
# --- Baseline (SVM seleccionado) ---
y_pred_train_svm = best_svm_model.predict(X_train_scaled)
y_pred_test_svm = best_svm_model.predict(X_test_scaled)

y_prob_train_svm = best_svm_model.predict_proba(X_train_scaled)[:,1]
y_prob_test_svm = best_svm_model.predict_proba(X_test_scaled)[:,1]

print("\n--- Mejor SVM (baseline) ---")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train_svm))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test_svm))
print("Train AUC:", roc_auc_score(y_train, y_prob_train_svm))
print("Test AUC:", roc_auc_score(y_test, y_prob_test_svm))

# --- Bagging con tuning ---
param_grid_bagging = {
    "n_estimators": [10, 30, 50, 100],
    "max_samples": [0.5, 0.7, 1.0],
    "max_features": [0.5, 0.7, 1.0],
    "bootstrap": [True],
    "bootstrap_features": [False]
}

bagging = BaggingClassifier(estimator=best_svm_model,
                            random_state=SEMILLA, n_jobs=-1)

grid_bagging = GridSearchCV(
    bagging,
    param_grid=param_grid_bagging,
    scoring={'acc':'accuracy','auc':'roc_auc'},
    refit='auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_bagging.fit(X_train_scaled, y_train)
print("\n--- Bagging tuning ---")
print("Mejores parámetros:", grid_bagging.best_params_)
print("Mejor AUC CV:", grid_bagging.best_score_)

best_bagging = grid_bagging.best_estimator_

# --- Evaluación comparativa ---
def eval_model(name, model, Xtr, ytr, Xte, yte):
    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)
    ytr_prob = model.predict_proba(Xtr)[:,1]
    yte_prob = model.predict_proba(Xte)[:,1]
    acc_tr = accuracy_score(ytr, ytr_pred)
    acc_te = accuracy_score(yte, yte_pred)
    auc_tr = roc_auc_score(ytr, ytr_prob)
    auc_te = roc_auc_score(yte, yte_prob)
    return {"Modelo": name, "Acc Train": acc_tr, "Acc Test": acc_te,
            "AUC Train": auc_tr, "AUC Test": auc_te}

res_base = eval_model("SVM base", best_svm_model, X_train_scaled, y_train, X_test_scaled, y_test)
res_bagg = eval_model("Bagging SVM", best_bagging, X_train_scaled, y_train, X_test_scaled, y_test)

comparacion = pd.DataFrame([res_base, res_bagg])
print("\nComparación SVM vs Bagging (con tuning):\n", comparacion)

# --- Matrices de confusión ---
cm_svm = confusion_matrix(y_test, best_svm_model.predict(X_test_scaled))
print("\nMatriz de confusion SVM base")
print(cm_svm)
cm_bag = confusion_matrix(y_test, best_bagging.predict(X_test_scaled))
print("\nMatriz de confusion bagging")
print(cm_bag)

fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Black','White'], yticklabels=['Black','White'], ax=axes[0])
axes[0].set_title("Confusión – SVM base")

sns.heatmap(cm_bag, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Black','White'], yticklabels=['Black','White'], ax=axes[1])
axes[1].set_title("Confusión – Bagging SVM")

plt.tight_layout(); plt.show()

###############################################################################
# PUNTO 4: Stacking en profundidad (base learners diversos + meta-learner)
###############################################################################
# Definimos modelos base
base_models = [
    ('rf', RandomForestClassifier(random_state=SEMILLA, n_estimators=200, max_depth=None)),
    ('lr', LogisticRegression(random_state=SEMILLA, max_iter=1000)),
    ('knn', KNeighborsClassifier(n_neighbors=15)),
    ('svm', best_svm_model)  # incluimos el mejor SVM como uno más
]

# Meta-modelo (lineal para evitar sobreajuste del RF)
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(random_state=SEMILLA, max_iter=1000),
    cv=5,
    passthrough=True,
    n_jobs=-1
)

# Entrenamos y evaluamos
stacking_model.fit(X_train_scaled, y_train)
y_pred_train_stack = stacking_model.predict(X_train_scaled)
y_pred_test_stack = stacking_model.predict(X_test_scaled)

y_prob_train_stack = stacking_model.predict_proba(X_train_scaled)[:,1]
y_prob_test_stack = stacking_model.predict_proba(X_test_scaled)[:,1]

print("\n--- STACKING (con LR meta, passthrough) ---")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train_stack))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test_stack))
print("Train AUC:", roc_auc_score(y_train, y_prob_train_stack))
print("Test AUC:", roc_auc_score(y_test, y_prob_test_stack))

# ========================
# Evaluación individual de bases
# ========================
results_stack = []
for name, model in base_models:
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)
    y_prob_test = model.predict_proba(X_test_scaled)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test_scaled)
    results_stack.append({
        "Modelo": f"Base–{name.upper()}",
        "Acc Train": accuracy_score(y_train, model.predict(X_train_scaled)),
        "Acc Test": accuracy_score(y_test, y_pred_test),
        "AUC Train": roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:,1]) if hasattr(model, "predict_proba") else None,
        "AUC Test": roc_auc_score(y_test, y_prob_test)
    })
    print(f"\nMatriz de confusión – {name.upper()} (TEST):")
    print(confusion_matrix(y_test, y_pred_test))

# ========================
# Resultados del stacking
# ========================
results_stack.append({
    "Modelo": "STACKING (LR meta)",
    "Acc Train": accuracy_score(y_train, y_pred_train_stack),
    "Acc Test": accuracy_score(y_test, y_pred_test_stack),
    "AUC Train": roc_auc_score(y_train, y_prob_train_stack),
    "AUC Test": roc_auc_score(y_test, y_prob_test_stack)
})
print("\nMatriz de confusión – STACKING (TEST):")
print(confusion_matrix(y_test, y_pred_test_stack))

df_results_stack = pd.DataFrame(results_stack)
print("\n=== Resultados comparativos (Stacking y bases) ===")
print(df_results_stack)

# ========================
# Comparativa global con SVM, Bagging y Stacking
# ========================
comparativa_final = pd.concat([
    comparacion,  # lo que ya tenías de SVM y Bagging
    df_results_stack
], ignore_index=True)

# >>> 1) Añadir gaps train-test
comparativa_final["Acc Gap"] = comparativa_final["Acc Train"] - comparativa_final["Acc Test"]
comparativa_final["AUC Gap"] = comparativa_final["AUC Train"] - comparativa_final["AUC Test"]

# >>> 2) Correlación de predicciones en stacking
preds_stack = pd.DataFrame({
    name: model.fit(X_train_scaled, y_train).predict(X_test_scaled)
    for name, model in base_models
})
print("\n=== Correlación entre predicciones de clasificadores base (TEST) ===")
print(preds_stack.corr())

# >>> 3) Añadir Precision, Recall y F1
from sklearn.metrics import precision_score, recall_score, f1_score

def add_extra_metrics(df, Xte, yte):
    metrics = []
    for i, row in df.iterrows():
        model_name = row["Modelo"]
        if model_name == "SVM base":
            model = best_svm_model
        elif model_name == "Bagging SVM":
            model = best_bagging
        elif model_name == "Stacking (LR meta)":
            model = stacking_model
        """elif "Base–RF" in model_name:
            model = base_models[0][1]
        elif "Base–LR" in model_name:
            model = base_models[1][1]
        elif "Base–KNN" in model_name:
            model = base_models[2][1]
        elif "Base–SVM" in model_name:
            model = base_models[3][1]"""
        else:
            continue
        y_pred = model.predict(Xte)
        metrics.append({
            "Modelo": model_name,
            "Precision": precision_score(yte, y_pred),
            "Recall": recall_score(yte, y_pred),
            "F1": f1_score(yte, y_pred)
        })
    return pd.DataFrame(metrics)

extra_metrics = add_extra_metrics(comparativa_final, X_test_scaled, y_test)
comparativa_final = comparativa_final.merge(extra_metrics, on="Modelo")

print("\n=== Comparativa FINAL con métricas adicionales ===")
print(comparativa_final.to_string(index=False))

# >>> 4) Gráfico comparativo de F1 en Test
plt.figure(figsize=(8,6))
sns.barplot(data=comparativa_final, x="Modelo", y="F1", palette="coolwarm")
plt.title("Comparación de F1-score en Test")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
