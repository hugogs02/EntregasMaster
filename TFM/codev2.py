import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.ensemble import StackingRegressor
from sklearn.base import clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Conv1D, Flatten, GRU
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# ---------------- LOAD & WEEKLY PIVOT ----------------
def load_and_pivot(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Company','Date']).reset_index(drop=True)
    df_pivot = df.pivot(index='Date', columns='Company', values='Close')
    df_weekly = df_pivot.resample('W').last()
    df_weekly = df_weekly.fillna(method='ffill').fillna(0)
    return df_weekly

# ---------------- CREATE SUPERVISED DATA ----------------
def create_supervised(df, W=30, H=10):
    X, y = [], []
    for col in df.columns:
        data = df[col].values
        for i in range(len(data) - W - H + 1):
            X.append(data[i:i+W])
            y.append(data[i+W:i+W+H])
    return np.array(X), np.array(y)

# ---------------- METRICS ----------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    da = np.mean((y_true > 0) == (y_pred > 0))
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))
    evs = explained_variance_score(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "DA": da,
        "SMAPE": smape,
        "EVS": evs
    }

# ---------------- DEEP LEARNING BUILDERS ----------------
def build_lstm(input_shape, units=50, dropout=0.3, H=10):
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_rnn(input_shape, units=50, dropout=0.3, H=10):
    model = Sequential([
        SimpleRNN(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_conv1d(input_shape, filters=32, kernel_size=3, dropout=0.3, H=10, **kwargs):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        Flatten(),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape, units=50, dropout=0.3, H=10, **kwargs):
    model = Sequential([
        GRU(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------- EVALUATION ----------------
def eval_ml_tscv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if y.shape[1] > 1:
            model_fit = MultiOutputRegressor(model)
        else:
            model_fit = model
        model_fit.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

def eval_dl_tscv(model_builder, X, y, units=50, dropout=0.3, epochs=20, batch_size=64, n_splits=3, **kwargs):
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for train_idx, test_idx in tscv.split(X_seq):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = model_builder(input_shape=(X_train.shape[1],1), units=units, dropout=dropout, H=y_train.shape[1], **kwargs)
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
        y_pred = model.predict(X_test, verbose=0)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

# ---------------- STACKING CON TIME SERIES SPLIT ----------------
def eval_stacking_tscv(base_models, meta_model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    S_train = np.zeros((X.shape[0], len(base_models)))
    S_test_all = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for i, (name, model) in enumerate(base_models):
            model_fold = clone(model)
            if y.shape[1] > 1:
                model_fold = MultiOutputRegressor(model_fold)
            model_fold.fit(X_train, y_train)
            S_train[test_idx, i] = model_fold.predict(X_test).mean(axis=1)
        S_test_all.append(S_train[test_idx, :])

    meta_model_fit = clone(meta_model)
    if y.shape[1] > 1:
        meta_model_fit = MultiOutputRegressor(meta_model_fit)
    meta_model_fit.fit(S_train, y)

    S_test_all = np.vstack(S_test_all)
    y_pred = meta_model_fit.predict(S_train)

    return compute_metrics(y, y_pred), meta_model_fit

# ---------------- MAIN ----------------
df_weekly = load_and_pivot('stock_details_5_years.csv')
W, H = 25, 10
X, y = create_supervised(df_weekly, W=W, H=H)

X = np.nan_to_num(X)
y = np.nan_to_num(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- XGB & LGBM ----
xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
lgb_params = {"n_estimators":[100,200], "max_depth":[5,10], "learning_rate":[0.01,0.05]}
results = {}

best_score, best_model = -np.inf, None
for params in ParameterGrid(xgb_params):
    print(f"XGB with {params}")
    model = XGBRegressor(random_state=42, verbosity=0, **params)
    metrics = eval_ml_tscv(model, X_scaled, y)
    if metrics["R2"] > best_score:
        best_score = metrics["R2"]
        best_model = model
results["XGB"] = eval_ml_tscv(best_model, X_scaled, y)

best_score, best_model = -np.inf, None
for params in ParameterGrid(lgb_params):
    print(f"LGBM with {params}")
    model = LGBMRegressor(random_state=42, verbose=-1, **params)
    metrics = eval_ml_tscv(model, X_scaled, y)
    if metrics["R2"] > best_score:
        best_score = metrics["R2"]
        best_model = model
results["LGBM"] = eval_ml_tscv(best_model, X_scaled, y)

# ---- DEEP LEARNING ----
print("LSTM")
results["LSTM"] = eval_dl_tscv(build_lstm, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)

print("RNN")
results["RNN"] = eval_dl_tscv(build_rnn, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)

print("Conv1D")
results["Conv1D"] = eval_dl_tscv(build_conv1d, X_scaled, y, filters=32, kernel_size=3, dropout=0.3, epochs=20, batch_size=64)

print("GRU")
results["GRU"] = eval_dl_tscv(build_gru, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)

# ---------------- CONVERTIR A DATAFRAME ----------------
df_results = pd.DataFrame([
    {"model": k, **v} for k, v in results.items()
])

# ---------------- IDENTIFICAR LOS MEJORES ----------------
best_models = {}
for metric in ["R2", "RMSE", "MAE", "SMAPE", "EVS"]:
    if metric in ["RMSE", "MAE", "SMAPE"]:
        best_models[metric] = df_results.loc[df_results[metric].idxmin()]
    else:
        best_models[metric] = df_results.loc[df_results[metric].idxmax()]

# ---------------- IMPRIMIR RESULTADOS ----------------
print("\n--- RESULTADOS COMPLETOS ---")
print(df_results.sort_values(by="R2", ascending=False).reset_index(drop=True))

print("\n--- MEJORES MODELOS POR MÉTRICA ---")
for metric, row in best_models.items():
    print(f"{metric}: {row['model']} ({metric}={row[metric]:.3f})")

# ---------------- VISUALIZACIÓN ----------------
plt.figure(figsize=(12,6))
metrics_to_plot = ["R2", "RMSE", "MAE", "SMAPE", "EVS"]
df_plot = df_results.melt(id_vars=["model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Value")

sns.barplot(data=df_plot, x="model", y="Value", hue="Metric")
plt.title("Comparación de modelos por métricas")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()