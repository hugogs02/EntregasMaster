####### WORKS GOOD, TRYING TO IMPROVE
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

# --- SECCIÓN MODIFICADA PARA REDUCIR OVERFITTING ---

def eval_dl_tscv(model_builder, X, y, units=32, dropout=0.4, epochs=50, batch_size=64, n_splits=3, **kwargs):
    """
    Cross-validation para modelos DL con EarlyStopping y validación interna.
    """
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    histories = []
    preds_last_fold, y_test_last_fold = None, None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_seq)):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_builder(
            input_shape=(X_train.shape[1],1),
            units=units,
            dropout=dropout,
            H=y_train.shape[1],
            **kwargs
        )

        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[es]
        )
        histories.append(history.history)

        y_pred = model.predict(X_test, verbose=0)
        metrics_list.append(compute_metrics(y_test, y_pred))

        # Guardamos última predicción para graficar
        preds_last_fold, y_test_last_fold = y_pred, y_test

    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return avg_metrics, histories, preds_last_fold, y_test_last_fold


# --- MAIN (EJEMPLO DE USO) ---
df_weekly = load_and_pivot('stock_details_5_years.csv')
W, H = 25, 4
X, y = create_supervised(df_weekly, W=W, H=H)

X = np.nan_to_num(X)
y = np.nan_to_num(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}
histories_dict = {}
preds_dict = {}
ytest_dict = {}

# LSTM
print("LSTM")
metrics, histories, preds, ytest = eval_dl_tscv(build_lstm, X_scaled, y)
results["LSTM"] = metrics
histories_dict["LSTM"] = histories
preds_dict["LSTM"] = preds
ytest_dict["LSTM"] = ytest

# GRU
print("GRU")
metrics, histories, preds, ytest = eval_dl_tscv(build_gru, X_scaled, y)
results["GRU"] = metrics
histories_dict["GRU"] = histories
preds_dict["GRU"] = preds
ytest_dict["GRU"] = ytest

# Conv1D
print("Conv1D")
metrics, histories, preds, ytest = eval_dl_tscv(build_conv1d, X_scaled, y)
results["Conv1D"] = metrics
histories_dict["Conv1D"] = histories
preds_dict["Conv1D"] = preds
ytest_dict["Conv1D"] = ytest

# ---------------- VISUALIZACIÓN ----------------
df_results = pd.DataFrame([{"model": k, **v} for k, v in results.items()])
print(df_results)

# --- Gráfico de métricas ---
plt.figure(figsize=(12,6))
metrics_to_plot = ["R2","RMSE","MAE","SMAPE","EVS"]
df_plot = df_results.melt(id_vars=["model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Value")
sns.barplot(data=df_plot, x="model", y="Value", hue="Metric")
plt.title("Comparación de modelos DL con regularización")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Gráfico train vs val loss de un modelo (ej: LSTM) ---
plt.figure(figsize=(8,5))
plt.plot(histories_dict["LSTM"][0]['loss'], label='Train loss')
plt.plot(histories_dict["LSTM"][0]['val_loss'], label='Val loss')
plt.title("Evolución del loss (LSTM, fold 1)")
plt.legend()
plt.show()

# --- Comparar predicciones vs reales en 10 empresas (último fold) ---
idx_sample = np.random.choice(range(ytest_dict["GRU"].shape[0]), size=10, replace=False)
plt.figure(figsize=(12,6))
for i, idx in enumerate(idx_sample):
    plt.plot(ytest_dict["LSTM"][idx], label=f"Real {i}")
    plt.plot(preds_dict["LSTM"][idx], '--', label=f"Pred {i}")
    plt.title("Predicciones vs reales en submuestra de 10 empresas (LSTM)")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()
