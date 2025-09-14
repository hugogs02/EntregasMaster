import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.base import clone

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from itertools import product

SEED = 42
np.random.seed(SEED)

# =========================================================
# LOAD & PIVOT
# =========================================================
def load_and_pivot(path, max_nan_ratio=0.4):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Company','Date']).reset_index(drop=True)
    df_pivot = df.pivot(index='Date', columns='Company', values='Close')
    mask_keep = df_pivot.isna().mean() <= max_nan_ratio
    df_pivot = df_pivot.loc[:, mask_keep]
    df_pivot = df_pivot.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
    return df_pivot.resample('W').last()

# =========================================================
# CREATE SUPERVISED
# =========================================================
def create_supervised_with_dates(df, W=25, H=1):
    X, y, dates = [], [], []
    for col in df.columns:
        data = df[col].values
        idx = df.index
        for i in range(len(data) - W - H + 1):
            X.append(data[i:i+W])
            y.append(data[i+W:i+W+H])  
            dates.append(idx[i+W:i+W+H])
    return np.array(X), np.array(y), np.array(dates)

# =========================================================
# METRICS
# =========================================================
def compute_metrics(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-6))) * 100
    smape = 100*np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-6))
    evs = explained_variance_score(y_true, y_pred)
    return {"MSE":mse,"RMSE":rmse,"MAE":mae,"R2":r2,"MAPE":mape,"SMAPE":smape,"EVS":evs}

# =========================================================
# DL BUILDERS
# =========================================================
def build_lstm(input_shape, units=16, dropout=0.3, H=1, **kwargs):
    model = Sequential([LSTM(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape, units=16, dropout=0.3, H=1, **kwargs):
    model = Sequential([GRU(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_conv1d(input_shape, units=16, dropout=0.3, kernel_size=3, H=1, **kwargs):
    model = Sequential([Conv1D(filters=units, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
                        Flatten(),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

# =========================================================
# TSCV: ML
# =========================================================
def tscv_ml(model, X, y, n_splits=5):
    X_flat = X.reshape((X.shape[0], -1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics = []
    for train_idx, test_idx in tscv.split(X_flat):
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_fold = MultiOutputRegressor(clone(model)) if y.shape[1] > 1 else clone(model)
        model_fold.fit(X_train_scaled, y_train)
        y_pred = model_fold.predict(X_test_scaled)
        all_metrics.append(compute_metrics(y_test, y_pred))
    return pd.DataFrame(all_metrics)

# =========================================================
# TSCV: DL
# =========================================================
def tscv_dl(model_builder, X, y, n_splits=3, units=16, dropout=0.3, epochs=10, batch_size=32, kernel_size=3):
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics = []
    for train_idx, test_idx in tscv.split(X_seq):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_2d = X_train.reshape((X_train.shape[0], -1))
        X_test_2d = X_test.reshape((X_test.shape[0], -1))
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

        model = model_builder(input_shape=(X_train.shape[1],1),
                              units=units, dropout=dropout, H=y_train.shape[1], kernel_size=kernel_size)
        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
        model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                  epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es, lr])
        y_pred = model.predict(X_test_scaled, verbose=0)
        all_metrics.append(compute_metrics(y_test, y_pred))
    return pd.DataFrame(all_metrics)

# =========================================================
# RUN EXPERIMENT (train + validation)
# =========================================================
def run_experiment(path_csv, W=25, H=1, sub_sample_ratio=0.3):
    df_weekly = load_and_pivot(path_csv)
    sampled_cols = np.random.choice(df_weekly.columns, int(len(df_weekly.columns)*sub_sample_ratio), replace=False)
    df_sub = df_weekly[sampled_cols]

    X_sub, y_sub, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X_sub, y_sub = np.nan_to_num(X_sub), np.nan_to_num(y_sub)

    ml_models = {"XGB": XGBRegressor(random_state=SEED, verbosity=0, n_estimators=100, max_depth=3, learning_rate=0.05),
                 "LGBM": LGBMRegressor(random_state=SEED, verbose=-1, n_estimators=100, max_depth=5, learning_rate=0.05)}
    dl_models = {"LSTM": build_lstm, "GRU": build_gru, "Conv1D": build_conv1d}

    results, best_models = {}, {}

    # ML
    for name, model in ml_models.items():
        df_metrics = tscv_ml(model, X_sub, y_sub, n_splits=3)
        results[name] = df_metrics["R2"].mean()
        best_models[name] = model

    # DL
    configs = {"LSTM":{"units":16,"dropout":0.3}, "GRU":{"units":16,"dropout":0.3}, "Conv1D":{"units":16,"dropout":0.3,"kernel_size":3}}
    for name, builder in dl_models.items():
        df_metrics = tscv_dl(builder, X_sub, y_sub, n_splits=3, **configs[name])
        results[name] = df_metrics["R2"].mean()
        best_models[name] = configs[name]

    return results, best_models, df_sub

# =========================================================
# MAIN PIPELINE
# =========================================================
if __name__=="__main__":
    path_csv = "stock_details_5_years.csv"

    print("\n=== Experimento H=1 ===")
    res1, models1, df_sub1 = run_experiment(path_csv, H=1)
    print(res1)

    print("\n=== Experimento H=4 ===")
    res4, models4, df_sub4 = run_experiment(path_csv, H=4)
    print(res4)

    print("\nComparación R2 medio por modelo:")
    for k in set(res1.keys()) | set(res4.keys()):
        print(f"{k}: H=1 → {res1.get(k,'-'):.4f}, H=4 → {res4.get(k,'-'):.4f}")

    # ---------------- GRÁFICOS COMPARATIVOS ----------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Top 6 por H=4
    n_companies = df_sub4.shape[1]
    last_values = df_sub4.iloc[-1].values
    # simulamos predicciones finales -> aquí deberías meter las de Bagging
    y_pred_last_H4 = last_values * (1 + 0.05*np.random.randn(n_companies,4))  
    y_pred_last_H1 = last_values * (1 + 0.05*np.random.randn(n_companies))  

    diff_abs = y_pred_last_H4[:,-1] - last_values
    top6_idx = np.argsort(diff_abs)[-6:][::-1]

    for j, idx in enumerate(top6_idx):
        company = df_sub4.columns[idx]
        series = df_sub4[company]

        axes[j].plot(series.index, series.values, color="black", label="Histórico")

        # H=4 predicciones (rojo)
        future_dates4 = [series.index[-1] + pd.Timedelta(weeks=k) for k in range(1,5)]
        axes[j].plot([series.index[-1]]+future_dates4, [series.values[-1]]+list(y_pred_last_H4[idx]), "ro-", label="H=4")

        # H=1 predicción (azul)
        future_date1 = series.index[-1] + pd.Timedelta(weeks=1)
        axes[j].plot([series.index[-1], future_date1], [series.values[-1], y_pred_last_H1[idx]], "bo--", label="H=1")

        axes[j].set_title(company)
        axes[j].legend()
        axes[j].grid(True)

    plt.tight_layout()
    plt.show()
