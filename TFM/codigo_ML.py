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

dl_models = {"LSTM": build_lstm, "GRU": build_gru, "Conv1D": build_conv1d}

# =========================================================
# TSCV: ML
# =========================================================
def tscv_ml(model, X, y, n_splits=5):
    X_flat = X.reshape((X.shape[0], -1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for train_idx, test_idx in tscv.split(X_flat):
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_fold = MultiOutputRegressor(clone(model)) if y.shape[1] > 1 else clone(model)
        model_fold.fit(X_train_scaled, y_train if y_train.shape[1] > 1 else np.ravel(y_train))
        y_pred = model_fold.predict(X_test_scaled)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

# =========================================================
# TSCV: DL
# =========================================================
def tscv_dl(model_builder, X, y, n_splits=3, units=16, dropout=0.3, epochs=10, batch_size=32, kernel_size=3):
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
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
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

# =========================================================
# RUN EXPERIMENT (H=1 o H=4)
# =========================================================
def run_experiment(path_csv, W=25, H=1, sub_sample_ratio=0.3):
    np.random.seed(SEED)
    df_weekly = pd.read_csv(path_csv, index_col=0, parse_dates=True)
    df_weekly.index = pd.to_datetime(df_weekly.index, utc=True)
    sampled_cols = np.random.choice(df_weekly.columns, int(len(df_weekly.columns)*sub_sample_ratio), replace=False)
    df_sub = df_weekly[sampled_cols]

    X_sub, y_sub, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X_sub, y_sub = np.nan_to_num(X_sub), np.nan_to_num(y_sub)

    xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
    lgb_params = {"n_estimators":[100,200], "max_depth":[5,10], "learning_rate":[0.01,0.05]}
    dl_params = {'units':[16,32], 'dropout':[0.2,0.3]}
    c1d_params = {'units':[16,32], 'dropout':[0.2,0.3], 'kernel_size':[3,5]}

    ml_models = {"XGB": XGBRegressor, "LGBM": LGBMRegressor}
    results, best_models = {}, {}

    # ML
    for name, ModelClass in ml_models.items():
        best_score, best_model = -np.inf, None
        param_grid = xgb_params if name=="XGB" else lgb_params
        for params in ParameterGrid(param_grid):
            print(f"\nEntrenando {name} con {params}")
            model = ModelClass(random_state=SEED, verbosity=0, **params) if name=="XGB" else ModelClass(random_state=SEED, verbose=-1, **params)
            metrics = tscv_ml(model, X_sub, y_sub, n_splits=3)
            print(f"R2={metrics['R2']:.4f}")
            if metrics['R2'] > best_score:
                best_score, best_model = metrics['R2'], model
        results[name] = best_score
        best_models[name] = best_model

    # DL
    for name, builder in dl_models.items():
        best_score, best_config = -np.inf, None
        combos = product(c1d_params['units'], c1d_params['dropout'], c1d_params['kernel_size']) if name=="Conv1D" else product(dl_params['units'], dl_params['dropout'])
        for combo in combos:
            config = {'units':combo[0], 'dropout':combo[1]} if name!="Conv1D" else {'units':combo[0],'dropout':combo[1],'kernel_size':combo[2]}
            print(f"\nEntrenando {name} con {config}")
            metrics = tscv_dl(builder, X_sub, y_sub, n_splits=3, **config)
            print(f"R2={metrics['R2']:.4f}")
            if metrics['R2'] > best_score:
                best_score, best_config = metrics['R2'], config
        results[name] = best_score
        best_models[name] = best_config

    return results, best_models, df_sub

# =========================================================
# BAGGING + FORECAST
# =========================================================
# =========================================================
def run_bagging_and_forecast(df_sub, W=25, H=1, best_models=None, top_models=None, n_bags=5):
    """
    Hace Bagging con los mejores modelos y predice en el último fold + futuro.
    """
    if top_models is None:
        top_models = ["XGB", "LGBM", "Conv1D"]

    # --- Datos supervisados ---
    X, y, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X, y = np.nan_to_num(X), np.nan_to_num(y)
    X_flat = X.reshape((X.shape[0], -1))

    # --- TimeSeriesSplit ---
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X_flat))[-1]
    X_train, X_test = X_flat[train_idx], X_flat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Escalado
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Bagging ---
    bagging_preds = np.zeros_like(y_test, dtype=float)

    for name in top_models:
        preds_bag = np.zeros_like(y_test, dtype=float)
        for _ in range(n_bags):
            if name in ["XGB", "LGBM"]:
                base = clone(best_models[name])
                # MultiOutput wrapper si H>1
                if H > 1:
                    model = MultiOutputRegressor(base)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                else:
                    model = base
                    model.fit(X_train, np.ravel(y_train))
                    preds = model.predict(X_test).reshape(-1, 1)
            else:
                # DL models
                X_train_seq = X_train.reshape((X_train.shape[0], W, 1))
                X_test_seq = X_test.reshape((X_test.shape[0], W, 1))
                config = best_models[name]
                model = dl_models[name](input_shape=(W,1), H=H, **config)
                es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
                model.fit(X_train_seq, y_train,
                          validation_data=(X_test_seq, y_test),
                          epochs=20, batch_size=32, verbose=0, callbacks=[es, lr])
                preds = model.predict(X_test_seq, verbose=0)

            preds_bag += preds

        preds_bag /= n_bags
        bagging_preds += preds_bag

    bagging_preds /= len(top_models)

    # --- Métricas ---
    metrics = compute_metrics(y_test, bagging_preds)
    print(f"Métricas Bagging (H={H}):", {k: round(v,4) for k,v in metrics.items()})

    # --- Reentreno con todos los datos ---
    X_full, y_full, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X_full, y_full = np.nan_to_num(X_full), np.nan_to_num(y_full)
    X_full_flat = X_full.reshape((X_full.shape[0], -1))
    X_full_scaled = scaler.fit_transform(X_full_flat)

    trained_models = {}
    for name in top_models:
        if name in ["XGB", "LGBM"]:
            base = clone(best_models[name])
            if H > 1:
                model = MultiOutputRegressor(base)
                model.fit(X_full_scaled, y_full)
            else:
                base.fit(X_full_scaled, np.ravel(y_full))
                model = base
        else:
            X_full_seq = X_full_scaled.reshape((X_full_scaled.shape[0], W, 1))
            config = best_models[name]
            model = dl_models[name](input_shape=(W,1), H=H, **config)
            es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
            model.fit(X_full_seq, y_full, epochs=20, batch_size=32, verbose=0, callbacks=[es, lr])
        trained_models[name] = model

    # --- Predicción futura (última ventana de cada empresa) ---
    n_companies = df_sub.shape[1]
    if H == 1:
        y_pred_last = np.zeros(n_companies)
    else:
        y_pred_last = np.zeros((n_companies, H))

    for i, company in enumerate(df_sub.columns):
        series = df_sub[company].values
        last_window = series[-W:].reshape(1, -1)
        last_window_scaled = scaler.transform(last_window)

        preds_models = []
        for name, model in trained_models.items():
            if name in ["XGB", "LGBM"]:
                preds_models.append(model.predict(last_window_scaled))
            else:
                last_seq = last_window_scaled.reshape((1, W, 1))
                preds_models.append(model.predict(last_seq, verbose=0))
        preds_models = np.array(preds_models)

        if H == 1:
            y_pred_last[i] = np.mean(preds_models)
        else:
            y_pred_last[i] = np.mean(preds_models.reshape(len(preds_models), -1, H), axis=0)

    return y_pred_last, metrics


# =========================================================
# PLOT FORECASTS
# =========================================================
def plot_forecasts_full(df_sub, y_pred_last_H1, y_pred_last_H4, H1=1, H4=4, title="Predicciones (Full)"):
    last_values = df_sub.iloc[-1].values
    
    # diferencias basadas en el último paso de H=4
    diff_abs = y_pred_last_H4[:,-1] - last_values
    top6_idx = np.argsort(diff_abs)[-6:][::-1]

    fig, axes = plt.subplots(3,2, figsize=(14,10))
    axes = axes.flatten()
    
    for j, idx in enumerate(top6_idx):
        company = df_sub.columns[idx]
        series = df_sub[company]
        last_date = series.index[-1]
        
        # histórico completo
        axes[j].plot(series.index, series.values, color="black", label="Histórico")
        
        # H=1
        next_date = last_date + pd.Timedelta(weeks=1)
        pred_val_H1 = float(y_pred_last_H1[idx])
        axes[j].plot([last_date, next_date],
                     [series.values[-1], pred_val_H1],
                     "bo--", label=f"Pred H={H1}")
        
        # H=4
        future_dates = [last_date + pd.Timedelta(weeks=k) for k in range(1,H4+1)]
        full_dates = [last_date] + future_dates
        full_preds = [series.values[-1]] + list(y_pred_last_H4[idx])
        axes[j].plot(full_dates, full_preds, "ro-", label=f"Pred H={H4}")
        
        axes[j].set_title(company)
        axes[j].legend()
        axes[j].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_forecasts_zoom(df_sub, y_pred_last_H1, y_pred_last_H4, H1=1, H4=4, title="Predicciones (Zoom últimas semanas)"):
    last_values = df_sub.iloc[-1].values
    
    # diferencias basadas en el último paso de H=4
    diff_abs = y_pred_last_H4[:,-1] - last_values
    top6_idx = np.argsort(diff_abs)[-6:][::-1]

    fig, axes = plt.subplots(3,2, figsize=(14,10))
    axes = axes.flatten()
    
    for j, idx in enumerate(top6_idx):
        company = df_sub.columns[idx]
        series = df_sub[company]
        last_date = series.index[-1]
        
        # último y penúltimo valor real
        axes[j].plot(series.index[-2:], series.values[-2:], "k-", label="Últimos reales", linewidth=2)
        
        # H=1
        next_date = last_date + pd.Timedelta(weeks=1)
        pred_val_H1 = float(y_pred_last_H1[idx])
        axes[j].plot([last_date, next_date],
                     [series.values[-1], pred_val_H1],
                     "bo--", label=f"Pred H={H1}")
        
        # H=4
        future_dates = [last_date + pd.Timedelta(weeks=k) for k in range(1,H4+1)]
        full_dates = [last_date] + future_dates
        full_preds = [series.values[-1]] + list(y_pred_last_H4[idx])
        axes[j].plot(full_dates, full_preds, "ro-", label=f"Pred H={H4}")
        
        axes[j].set_title(company)
        axes[j].legend()
        axes[j].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compare_forecasts(df_sub, y_pred_last_H1, y_pred_last_H4, H1=1, H4=4):
    """
    Ejecuta ambas visualizaciones:
    - plot_forecasts_full: muestra histórico completo + predicciones H=1 y H=4
    - plot_forecasts_zoom: muestra solo los últimos 2 valores reales + predicciones H=1 y H=4
    """
    print("\n=== Gráfico con histórico completo + predicciones ===")
    plot_forecasts_full(df_sub, y_pred_last_H1, y_pred_last_H4, H1=H1, H4=H4, title="Histórico completo con predicciones")

    print("\n=== Gráfico con zoom en últimos valores + predicciones ===")
    plot_forecasts_zoom(df_sub, y_pred_last_H1, y_pred_last_H4, H1=H1, H4=H4, title="Zoom últimos valores con predicciones")


# =========================================================
# MAIN
# =========================================================
if __name__=="__main__":
    path_csv = "stock_weekly_clean.csv"

    print("\n=== Experimento H=1 ===")
    res1, models1, df_sub1 = run_experiment(path_csv, H=1)
    y_pred_last_H1, metrics_H1 = run_bagging_and_forecast(df_sub1, W=25, H=1, best_models=models1)

    print("\n=== Experimento H=4 ===")
    res4, models4, df_sub4 = run_experiment(path_csv, H=4)
    y_pred_last_H4, metrics_H4 = run_bagging_and_forecast(df_sub4, W=25, H=4, best_models=models4)

    print("\nComparación R2 medio por modelo:")
    for k in set(res1.keys()) | set(res4.keys()):
        print(f"{k}: H=1 → {res1.get(k,'-')}, H=4 → {res4.get(k,'-')}")

    print("\nComparaciones y gráficos:")
    compare_forecasts(df_sub4, y_pred_last_H1, y_pred_last_H4, H1=1, H4=4)