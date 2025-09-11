import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

# ---------------- LOAD & WEEKLY AGGREGATION ----------------
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Company','Date']).reset_index(drop=True)
    df_weekly = []
    for _, g in df.groupby('Company'):
        g_weekly = g.set_index('Date').resample('W').last().reset_index()
        df_weekly.append(g_weekly)
    return pd.concat(df_weekly).reset_index(drop=True)

# ---------------- CREATE SUPERVISED DATA ----------------
def create_supervised(df, W=30, H=10, target_col="Close"):
    X, y = [], []
    data = df[target_col].values
    for i in range(len(data) - W - H + 1):
        X.append(data[i:i+W])
        y.append(data[i+W:i+W+H])
    return np.array(X), np.array(y)

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

# ---------------- EVALUATION FUNCTIONS ----------------
def eval_ml_tscv(model, X, y, n_splits=5):
    if y.ndim == 1:
        y = y.reshape(-1,1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_list, r2_list, da_list = [], [], []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if y.shape[1] > 1:
            model_fit = MultiOutputRegressor(model)
        else:
            model_fit = model

        model_fit.fit(X_train, y_train)
        y_pred = model_fit.predict(X_test)

        mse_list.append(mean_squared_error(y_test, y_pred))
        r2_list.append(r2_score(y_test, y_pred))
        da_list.append(np.mean((y_test>0) == (y_pred>0)))
    
    return {"MSE": np.mean(mse_list), "R2": np.mean(r2_list), "DA": np.mean(da_list)}

def eval_dl_tscv(model_builder, X, y, units=50, dropout=0.3, epochs=20, batch_size=64, n_splits=3):
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_list, r2_list, da_list = [], [], []

    for train_idx, test_idx in tscv.split(X_seq):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_builder(input_shape=(X_train.shape[1],1),
                              units=units, dropout=dropout, H=y_train.shape[1])
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
        y_pred = model.predict(X_test, verbose=0)

        mse_list.append(mean_squared_error(y_test, y_pred))
        r2_list.append(r2_score(y_test, y_pred))
        da_list.append(np.mean((y_test>0) == (y_pred>0)))
    
    return {"MSE": np.mean(mse_list), "R2": np.mean(r2_list), "DA": np.mean(da_list)}

# ---------------- MAIN PIPELINE ----------------
df = load_data('stock_details_5_years.csv')
W, H = 25, 10
X, y = create_supervised(df, W=W, H=H)

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- XGB & LGBM GridSearch ----
xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
lgb_params = {"n_estimators":[100,200], "max_depth":[5,10], "learning_rate":[0.01,0.05]}

results = {}

# --- XGBoost ---
best_score, best_model = -np.inf, None
for params in ParameterGrid(xgb_params):
    print(f"Probando XGBoost {params}")
    model = XGBRegressor(random_state=42, verbosity=0, **params)
    metrics = eval_ml_tscv(model, X_scaled, y)
    if metrics["R2"] > best_score:
        best_score = metrics["R2"]
        best_model = model
results["XGB"] = metrics

# --- LGBM ---
best_score, best_model = -np.inf, None
for params in ParameterGrid(lgb_params):
    print(f"Probando LGBM {params}")
    model = LGBMRegressor(random_state=42, verbose=-1, **params)
    metrics = eval_ml_tscv(model, X_scaled, y)
    if metrics["R2"] > best_score:
        best_score = metrics["R2"]
        best_model = model
results["LGBM"] = metrics

# --- LSTM ---
print("Probando LSTM")
results["LSTM"] = eval_dl_tscv(build_lstm, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)

# --- RNN ---
print("Probando RNN")
results["RNN"] = eval_dl_tscv(build_rnn, X_scaled, y, units=32, dropout=0.3, epochs=20, batch_size=64)

# --- STACKING META-MODELO (XGB+LGBM) MULTI-OUTPUT ---
xgb_final = XGBRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.01)
lgb_final = LGBMRegressor(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.01)

stack = StackingRegressor(
    estimators=[('xgb', xgb_final), ('lgbm', lgb_final)],
    final_estimator=XGBRegressor(random_state=42)
)

# Envolver en MultiOutputRegressor para poder predecir todos los H pasos
stack_multi = MultiOutputRegressor(stack)
stack_multi.fit(X_scaled, y)
y_stack_pred = stack_multi.predict(X_scaled)

# Evaluación
mse = mean_squared_error(y, y_stack_pred)
r2 = r2_score(y, y_stack_pred)
da = np.mean((y>0) == (y_stack_pred>0))

results["STACK"] = {"MSE": mse, "R2": r2, "DA": da}

# --- Mostrar resultados ---
for k,v in results.items():
    print(f"{k} - MSE={v['MSE']:.2f}, R2={v['R2']:.3f}, DA={v['DA']:.3f}")