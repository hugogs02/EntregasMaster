# full_pipeline_global.py
# Pipeline global (inspirado en el notebook de Kaggle) para:
# - cargar datos de muchas empresas
# - resample semanal (media)
# - crear ventanas W y horizonte H (multi-step)
# - entrenar modelos globales: XGBoost, LightGBM, LSTM, RNN
# - walk-forward evaluation (expanding window)
# - métricas: MSE, R2, MAE, DA, Sharpe, Volatility, MeanPred
# - gráficos comparativos
#
# Requisitos: pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Utils: carga y preparación
# -------------------------
def load_weekly_close(path, how='mean', date_col='Date', company_col='Company', close_col='Close'):
    """Carga CSV y resamplea semanalmente por company.
       how: 'mean' o 'last'"""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    df = df.sort_values([company_col, date_col]).reset_index(drop=True)
    out = []
    for name, g in df.groupby(company_col):
        g = g.set_index(date_col).sort_index()
        if how == 'mean':
            series = g[close_col].resample('W').mean()
        else:
            series = g[close_col].resample('W').last()
        s = series.dropna().to_frame().reset_index()
        s[company_col] = name
        out.append(s)
    if len(out) == 0:
        return pd.DataFrame(columns=[date_col, company_col, close_col])
    return pd.concat(out, ignore_index=True)[[date_col, company_col, close_col]]

def create_windows_for_company(df_company, W=25, H=10, target_col='Close'):
    """Crea X (n,W), y (n,H), dates_pred (n) para UNA empresa ordenada por Date"""
    data = df_company[target_col].values.astype(float)
    dates = df_company['Date'].values
    X, y, dates_pred = [], [], []
    for i in range(len(data) - W - H + 1):
        X.append(data[i:i+W])
        y.append(data[i+W:i+W+H])
        dates_pred.append(dates[i+W])   # fecha asociada a la predicción (primer paso)
    if len(X) == 0:
        return np.empty((0, W)), np.empty((0, H)), np.array([], dtype='datetime64[ns]')
    return np.array(X), np.array(y), np.array(dates_pred)

# -------------------------
# Model builders (defaults)
# -------------------------
def build_xgb():
    return XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=0.1, reg_lambda=1.0,
                        random_state=42, verbosity=0)

def build_lgbm():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                         num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1)

def build_lstm(input_shape, H, units=32, dropout=0.3):
    # input_shape: (W, n_features)
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_rnn(input_shape, H, units=32, dropout=0.3):
    model = Sequential([
        SimpleRNN(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(H)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------
# Walk-forward evaluation
# -------------------------
def walk_forward_eval_global(model_builder, X, y, W, H,
                             step=None, scaler_cls=StandardScaler,
                             is_dl=False, dl_fit_params=None, verbose=False):
    """
    X: (n_samples, W, n_features)  -- for DL it's 3D; for ML pass flattened features (n_samples, n_features_flat)
    y: (n_samples, H)
    model_builder: function returning estimator for ML (no args) OR builder for DL (input_shape, H)
    is_dl: if True, expects model_builder(input_shape=(W,n_features), H=H)
    scaler_cls: scaler class to instantiate per fold
    Returns dict with metrics and predictions aggregated
    """
    if X.shape[0] == 0:
        return None

    nsamples = X.shape[0]
    if step is None:
        step = max(1, H)

    initial_train = max(50, int(0.2 * nsamples))
    if initial_train >= nsamples:
        return None

    preds = []
    trues = []
    last_prices = []
    pred_dates_idx = []

    idx = initial_train
    while idx < nsamples:
        # training uses samples [0:idx), test sample is idx
        train_idx = slice(0, idx)
        test_idx = slice(idx, idx+1)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # scale: for DL we flatten features and scale per feature, then reshape
        scaler = scaler_cls()
        if is_dl:
            n_train = X_train.shape[0]
            W_ = X_train.shape[1]
            n_feat = X_train.shape[2]
            Xtr_flat = X_train.reshape(n_train, -1)
            Xte_flat = X_test.reshape(1, -1)
            Xtr_s = scaler.fit_transform(Xtr_flat)
            Xte_s = scaler.transform(Xte_flat)
            Xtr_dl = Xtr_s.reshape(n_train, W_, n_feat)
            Xte_dl = Xte_s.reshape(1, W_, n_feat)
            # build model
            model = model_builder(input_shape=(W_, n_feat), H=y_train.shape[1])
            # fit with early stopping
            es = EarlyStopping(monitor='val_loss', patience=dl_fit_params.get('patience', 5),
                               restore_best_weights=True, verbose=0)
            val_split = 0.1 if Xtr_dl.shape[0] > 20 else 0.0
            if val_split > 0:
                model.fit(Xtr_dl, y_train, epochs=dl_fit_params.get('epochs', 30),
                          batch_size=dl_fit_params.get('batch_size', 64),
                          validation_split=val_split, verbose=0, callbacks=[es])
            else:
                model.fit(Xtr_dl, y_train, epochs=dl_fit_params.get('epochs', 30),
                          batch_size=dl_fit_params.get('batch_size', 64), verbose=0)
            y_pred = model.predict(Xte_dl, verbose=0)  # (1,H)
            last_price = X_test[0, -1, 0]  # assume feature 0 is price
        else:
            # ML case: X provided already flattened (n_samples, n_features)
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)
            estimator = model_builder()
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                estimator = MultiOutputRegressor(estimator)
                estimator.fit(Xtr, y_train)
                y_pred = estimator.predict(Xte).reshape(1, -1)
            else:
                estimator.fit(Xtr, y_train.ravel())
                y_pred = estimator.predict(Xte).reshape(1, -1)
            # last price: if we assume the first W columns correspond to raw prices, last price is X_test[0, W-1]
            last_price = X_test[0, W-1] if X_test.ndim == 2 else X_test[0, -1]

        preds.append(y_pred.astype(float))
        trues.append(y_test.astype(float))
        last_prices.append(float(last_price))
        pred_dates_idx.append(idx)
        idx += step

    if len(preds) == 0:
        return None

    y_pred_all = np.vstack(preds)
    y_true_all = np.vstack(trues)
    last_prices = np.array(last_prices).reshape(-1)

    # Flatten metrics
    mse = mean_squared_error(y_true_all.ravel(), y_pred_all.ravel())
    r2 = r2_score(y_true_all.ravel(), y_pred_all.ravel())
    mae = mean_absolute_error(y_true_all.ravel(), y_pred_all.ravel())
    maxe = max_error(y_true_all.ravel(), y_pred_all.ravel())

    # directional accuracy using returns vs last price
    eps = 1e-9
    ret_true = (y_true_all - last_prices[:, None]) / (last_prices[:, None] + eps)
    ret_pred = (y_pred_all - last_prices[:, None]) / (last_prices[:, None] + eps)
    da = np.mean(np.sign(ret_true) == np.sign(ret_pred))

    meanpred = np.mean(ret_pred)
    vol = np.std(ret_pred)
    sharpe = (meanpred / (vol + 1e-12)) * np.sqrt(52)

    metrics = {'MSE': float(mse), 'R2': float(r2), 'MAE': float(mae), 'MaxError': float(maxe),
               'DA': float(da), 'MeanPred': float(meanpred), 'Volatility': float(vol), 'Sharpe': float(sharpe),
               'n_predictions': int(y_true_all.shape[0]), 'H': int(y_true_all.shape[1])}

    return {'metrics': metrics, 'y_true_all': y_true_all, 'y_pred_all': y_pred_all,
            'last_prices': last_prices, 'pred_idx': pred_dates_idx}

# -------------------------
# Preparar dataset GLOBAL
# -------------------------
def prepare_global_dataset(dfw, W=25, H=10, company_col='Company', target_col='Close', company_encoding='label'):
    """Crea dataset global concatenando ventanas de todas las empresas.
       company_encoding: 'label' -> LabelEncoder (int); we repeat it as constant feature for DL
    """
    le = LabelEncoder()
    dfw = dfw.copy()
    dfw['Company_enc'] = le.fit_transform(dfw[company_col].astype(str))

    X_blocks = []
    y_blocks = []
    companies_blocks = []
    dates_blocks = []

    for company, g in dfw.groupby(company_col):
        g = g.sort_values('Date').reset_index(drop=True)
        Xc, yc, dates_pred = create_windows_for_company(g, W=W, H=H, target_col=target_col)
        if Xc.shape[0] == 0:
            continue
        X_blocks.append(Xc)          # shape (m,W)
        y_blocks.append(yc)          # shape (m,H)
        companies_blocks.append(np.full((Xc.shape[0], 1), le.transform([company])[0]))
        dates_blocks.append(dates_pred)

    if len(X_blocks) == 0:
        raise ValueError("No companies have enough data with the chosen W/H")

    X = np.vstack(X_blocks)   # (N, W)
    y = np.vstack(y_blocks)   # (N, H)
    companies = np.vstack(companies_blocks).reshape(-1)  # (N,)
    # For ML models: flatten X and append company enc as extra column
    X_ml = np.hstack([X, companies.reshape(-1, 1)])  # shape (N, W+1)
    # For DL: create features (price feature + company feature repeated across time)
    company_feat = companies.reshape(-1, 1)
    company_repeated = np.repeat(company_feat, W, axis=1).reshape(-1, W, 1)
    X_dl = X.reshape(-1, W, 1)
    X_dl = np.concatenate([X_dl, company_repeated], axis=2)  # shape (N, W, 2)

    return X_ml, X_dl, y, companies, le

# -------------------------
# Visualización
# -------------------------
def plot_global_pred_vs_true(pred_dict, model_name, step_plot=200):
    """Dibuja primer paso del horizonte para predicciones vs reales (global)"""
    res = pred_dict
    if res is None:
        print(f"No results for {model_name}")
        return
    y_true = res['y_true_all'][:, 0]  # primer paso
    y_pred = res['y_pred_all'][:, 0]
    n = len(y_true)
    idxs = np.arange(n)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(idxs[:step_plot], y_true[:step_plot], label='Real (first H step)', lw=1)
    ax.plot(idxs[:step_plot], y_pred[:step_plot], label=f'{model_name} pred', lw=1)
    ax.set_title(f'Global: Real vs Pred (first horizon step) - {model_name}')
    ax.legend()
    plt.show()

# -------------------------
# Ejecución ejemplo
# -------------------------
if __name__ == '__main__':
    DATA_PATH = 'stock_details_5_years.csv'   # ajusta
    # parametros
    W = 25            # ventana en semanas
    H = 10            # horizonte en semanas (multi-step)
    dl_fit_params = {'epochs': 30, 'batch_size': 64, 'patience': 5}

    print("Loading and resampling weekly...")
    dfw = load_weekly_close(DATA_PATH, how='mean')    # usa mean para semanas
    print("Companies:", dfw['Company'].nunique(), "rows:", len(dfw))

    print("Preparing global dataset (this may take a moment)...")
    X_ml, X_dl, y, companies, le = prepare_global_dataset(dfw, W=W, H=H)
    print("Shapes: X_ml", X_ml.shape, "X_dl", X_dl.shape, "y", y.shape)

    # Standardize ML features (flattened)
    scaler_ml = StandardScaler().fit(X_ml)
    X_ml_scaled = scaler_ml.transform(X_ml)

    # For DL scale per flattened features too
    n, W_, nfeat = X_dl.shape
    X_dl_flat = X_dl.reshape(n, -1)
    scaler_dl = StandardScaler().fit(X_dl_flat)
    X_dl_scaled = scaler_dl.transform(X_dl_flat).reshape(n, W_, nfeat)

    results = {}
    print("\nRunning walk-forward for XGB (global)...")
    results['XGB'] = walk_forward_eval_global(lambda: build_xgb(), X_ml_scaled, y, W=W, H=H, is_dl=False)

    print("\nRunning walk-forward for LGBM (global)...")
    results['LGBM'] = walk_forward_eval_global(lambda: build_lgbm(), X_ml_scaled, y, W=W, H=H, is_dl=False)

    print("\nRunning walk-forward for LSTM (global)...")
    results['LSTM'] = walk_forward_eval_global(lambda input_shape, H: build_lstm(input_shape, H, units=32, dropout=0.3),
                                              X_dl_scaled, y, W=W, H=H, is_dl=True, dl_fit_params=dl_fit_params)

    print("\nRunning walk-forward for RNN (global)...")
    results['RNN'] = walk_forward_eval_global(lambda input_shape, H: build_rnn(input_shape, H, units=32, dropout=0.3),
                                             X_dl_scaled, y, W=W, H=H, is_dl=True, dl_fit_params=dl_fit_params)

    # Print metrics
    print("\n=== Global metrics ===")
    for k, v in results.items():
        if v is None:
            print(k, "=> no results")
        else:
            m = v['metrics']
            print(f"{k}: MSE={m['MSE']:.3f}, R2={m['R2']:.3f}, MAE={m['MAE']:.3f}, DA={m['DA']:.3f}, Sharpe={m['Sharpe']:.3f}")

    # Plots (first horizon)
    for k in ['XGB', 'LGBM', 'LSTM', 'RNN']:
        plot_global_pred_vs_true(results[k], k, step_plot=400)

    # You can also save results for per-company analysis or further backtesting
    # e.g. pd.DataFrame(results['XGB']['y_pred_all']).to_csv('xgb_preds.csv', index=False)
