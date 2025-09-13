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

# ---------------- LOAD & FILTER ----------------
def load_and_filter(path, max_nan_ratio=0.4, sample_ratio=0.3):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Company','Date']).reset_index(drop=True)
    df_pivot = df.pivot(index='Date', columns='Company', values='Close')

    # Filtrar empresas con más del 40% de nulos
    mask_keep = df_pivot.isna().mean() <= max_nan_ratio
    df_pivot = df_pivot.loc[:, mask_keep]

    # Interpolación temporal
    df_pivot = df_pivot.interpolate(method='time').fillna(0)

    # Submuestra aleatoria de empresas
    sampled_cols = np.random.choice(df_pivot.columns, int(len(df_pivot.columns)*sample_ratio), replace=False)
    df_sample = df_pivot[sampled_cols]

    # Resample semanal
    df_weekly = df_sample.resample('W').last()
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
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape, "DA": da, "SMAPE": smape, "EVS": evs}

# ---------------- DEEP LEARNING BUILDERS ----------------
def build_lstm(input_shape, units=50, dropout=0.3, H=10):
    model = Sequential([LSTM(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_rnn(input_shape, units=50, dropout=0.3, H=10):
    model = Sequential([SimpleRNN(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_conv1d(input_shape, filters=32, kernel_size=3, dropout=0.3, H=10, **kwargs):
    model = Sequential([Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
                        Flatten(),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape, units=50, dropout=0.3, H=10, **kwargs):
    model = Sequential([GRU(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------- EVALUATION ----------------
def eval_ml_tscv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model_fit = MultiOutputRegressor(model) if y.shape[1] > 1 else model
        model_fit.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

def eval_dl_tscv_history(model_builder, X, y, units=50, dropout=0.3, epochs=20, batch_size=64, n_splits=3, **kwargs):
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    histories = []
    for train_idx, test_idx in tscv.split(X_seq):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = model_builder(input_shape=(X_train.shape[1],1), units=units, dropout=dropout, H=y_train.shape[1], **kwargs)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=0, callbacks=[es])
        y_pred = model.predict(X_test, verbose=0)
        metrics_list.append(compute_metrics(y_test, y_pred))
        histories.append(history)
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return avg_metrics, histories

# ---------------- MAIN ----------------
df_weekly = load_and_filter('stock_details_5_years.csv', max_nan_ratio=0.4, sample_ratio=0.3)
W, H = 25, 4
X, y = create_supervised(df_weekly, W=W, H=H)
X = np.nan_to_num(X)
y = np.nan_to_num(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- ML ----
xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
lgb_params = {"n_estimators":[100,200], "max_depth":[5,10], "learning_rate":[0.01,0.05]}
results = {}
best_models = {}

# XGB
best_score, best_model = -np.inf, None
for params in ParameterGrid(xgb_params):
    print(f"XGB with {params}")
    model = XGBRegressor(random_state=42, verbosity=0, **params)
    metrics = eval_ml_tscv(model, X_scaled, y)
    if metrics["R2"] > best_score:
        best_score = metrics["R2"]
        best_model = model
results["XGB"] = eval_ml_tscv(best_model, X_scaled, y)
best_models["XGB"] = best_model

# LGBM
best_score, best_model = -np.inf, None
for params in ParameterGrid(lgb_params):
    print(f"LGBM with {params}")
    model = LGBMRegressor(random_state=42, verbose=-1, **params)
    metrics = eval_ml_tscv(model, X_scaled, y)
    if metrics["R2"] > best_score:
        best_score = metrics["R2"]
        best_model = model
results["LGBM"] = eval_ml_tscv(best_model, X_scaled, y)
best_models["LGBM"] = best_model

# ---- DL ----
results_dl, histories = {}, {}

print("LSTM")
results_dl["LSTM"], histories["LSTM"] = eval_dl_tscv_history(build_lstm, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)
print("RNN")
results_dl["RNN"], histories["RNN"] = eval_dl_tscv_history(build_rnn, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)
print("Conv1D")
results_dl["Conv1D"], histories["Conv1D"] = eval_dl_tscv_history(build_conv1d, X_scaled, y, filters=32, kernel_size=3, dropout=0.3, epochs=20, batch_size=64)
print("GRU")
results_dl["GRU"], histories["GRU"] = eval_dl_tscv_history(build_gru, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)

# ---------------- CONVERTIR A DATAFRAME ----------------
df_results = pd.DataFrame([{"model": k, **v} for k,v in {**results, **results_dl}.items()])

# ---------------- IMPRIMIR ----------------
print(df_results.sort_values(by="R2", ascending=False).reset_index(drop=True))

# ---------------- GRAFICO METRICAS ----------------
plt.figure(figsize=(12,6))
metrics_to_plot = ["R2","RMSE","MAE","SMAPE","EVS"]
df_plot = df_results.melt(id_vars=["model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Value")
sns.barplot(data=df_plot, x="model", y="Value", hue="Metric")
plt.title("Comparación de modelos por métricas")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

# ---------------- GRAFICOS TRAIN VS VAL LOSS DL ----------------
for model_name, history_list in histories.items():
    plt.figure(figsize=(8,4))
    for fold, history in enumerate(history_list):
        plt.plot(history.history['loss'], label=f'Train fold {fold+1}')
        plt.plot(history.history['val_loss'], label=f'Val fold {fold+1}', linestyle='--')
    plt.title(f'Train vs Val Loss - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

# ---------------- COMPARACION PREDICCIONES vs TEST (10 empresas) ----------------
sample_companies = np.random.choice(df_weekly.columns, 10, replace=False)
fig, axs = plt.subplots(5,2, figsize=(14,12))
axs = axs.flatten()

for i, comp in enumerate(sample_companies):
    # Crear datos supervisados para la empresa
    data = df_weekly[comp].values
    X_comp, y_comp = create_supervised(pd.DataFrame({comp:data}), W=W, H=H)
    X_comp = np.nan_to_num(X_comp)
    y_comp = np.nan_to_num(y_comp)
    X_comp_scaled = scaler.transform(X_comp)

    # Predecimos con el mejor ML (XGB como ejemplo)
    y_pred = best_models["XGB"].fit(X_scaled, y).predict(X_comp_scaled)

    axs[i].plot(range(len(y_comp[:,0])), y_comp[:,0], label="Test")
    axs[i].plot(range(len(y_pred[:,0])), y_pred[:,0], label="Pred")
    axs[i].set_title(comp)
    axs[i].legend()

plt.tight_layout()
plt.show()
