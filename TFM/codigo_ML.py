import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Conv1D, Flatten, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

    # Filtrar empresas con más del X% de nulos
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
def build_lstm(input_shape, units=32, dropout=0.5, H=10):
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_rnn(input_shape, units=32, dropout=0.5, H=10):
    model = Sequential([
        SimpleRNN(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_conv1d(input_shape, filters=32, kernel_size=3, dropout=0.5, H=10, **kwargs):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
        Flatten(),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape, units=32, dropout=0.5, H=10, **kwargs):
    model = Sequential([
        GRU(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------- EVALUATION (con escalado fold a fold) ----------------
def eval_ml_tscv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_fit = MultiOutputRegressor(model) if y.shape[1] > 1 else model
        model_fit.fit(X_train_scaled, y_train)
        y_pred = model_fit.predict(X_test_scaled)

        metrics_list.append(compute_metrics(y_test, y_pred))

    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

def eval_dl_tscv_history(model_builder, X, y, units=32, dropout=0.5, epochs=50,
                         batch_size=64, n_splits=3, **kwargs):

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    histories = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_seq = X_train_scaled.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_seq  = X_test_scaled.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = model_builder(input_shape=(X_train_seq.shape[1], 1),
                              units=units, dropout=dropout,
                              H=y_train.shape[1], **kwargs)

        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        history = model.fit(X_train_seq, y_train,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test_seq, y_test),
                            verbose=0,
                            callbacks=[es, rlrop])

        y_pred = model.predict(X_test_seq, verbose=0)
        metrics_list.append(compute_metrics(y_test, y_pred))
        histories.append(history)

    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return avg_metrics, histories

# ---------------- WALK-FORWARD SPLIT ----------------
def walk_forward_validation(model, X, y, window_size=500):
    """
    Entrena en una ventana creciente y predice el siguiente bloque.
    """
    metrics_list = []
    start = 0
    while start + window_size < len(X):
        end = start + window_size
        X_train, y_train = X[start:end], y[start:end]
        X_test, y_test = X[end:end+1], y[end:end+1]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_fit = MultiOutputRegressor(model) if y.shape[1] > 1 else model
        model_fit.fit(X_train_scaled, y_train)
        y_pred = model_fit.predict(X_test_scaled)

        metrics_list.append(compute_metrics(y_test, y_pred))
        start += 1

    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    return avg_metrics

# ---------------- MAIN ----------------
if __name__ == "__main__":
    df_weekly = load_and_filter('stock_details_5_years.csv', max_nan_ratio=0.4, sample_ratio=0.3)
    W, H = 25, 4
    X, y = create_supervised(df_weekly, W=W, H=H)
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    results = {}
    best_models = {}

    # ---- ML ----
    xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
    best_score, best_model = -np.inf, None
    for params in ParameterGrid(xgb_params):
        model = XGBRegressor(random_state=42, verbosity=0, **params)
        metrics = eval_ml_tscv(model, X, y)
        if metrics["R2"] > best_score:
            best_score, best_model = metrics["R2"], model
    results["XGB"] = eval_ml_tscv(best_model, X, y)
    best_models["XGB"] = best_model

    # ---- DL ----
    results_dl, histories = {}, {}
    results_dl["LSTM"], histories["LSTM"] = eval_dl_tscv_history(build_lstm, X, y)
    results_dl["GRU"], histories["GRU"] = eval_dl_tscv_history(build_gru, X, y)
    results_dl["RNN"], histories["RNN"] = eval_dl_tscv_history(build_rnn, X, y)
    results_dl["Conv1D"], histories["Conv1D"] = eval_dl_tscv_history(build_conv1d, X, y)

    # ---- RESULTS ----
    df_results = pd.DataFrame([{"model": k, **v} for k,v in {**results, **results_dl}.items()])
    print(df_results.sort_values(by="R2", ascending=False).reset_index(drop=True))

    # ---- METRICS PLOT ----
    plt.figure(figsize=(12,6))
    metrics_to_plot = ["R2","RMSE","MAE","SMAPE","EVS"]
    df_plot = df_results.melt(id_vars=["model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Value")
    sns.barplot(data=df_plot, x="model", y="Value", hue="Metric")
    plt.title("Comparación de modelos por métricas")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()
