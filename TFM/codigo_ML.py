import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import ParameterGrid
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# ---------------- LOAD & PIVOT ----------------
def load_and_pivot(path, max_nan_ratio=0.4):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Company','Date']).reset_index(drop=True)
    df_pivot = df.pivot(index='Date', columns='Company', values='Close')
    mask_keep = df_pivot.isna().mean() <= max_nan_ratio
    df_pivot = df_pivot.loc[:, mask_keep]
    df_pivot = df_pivot.interpolate(method='time').fillna(method='ffill').fillna(method='bfill')
    return df_pivot.resample('W').last()

# ---------------- CREATE SUPERVISED DATA ----------------
def create_supervised(df, W=25, H=4):
    X, y = [], []
    for col in df.columns:
        data = df[col].values
        for i in range(len(data)-W-H+1):
            X.append(data[i:i+W])
            y.append(data[i+W:i+W+H])
    return np.array(X), np.array(y)

# ---------------- METRICS ----------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-6))) * 100
    da = np.mean((y_true>0) == (y_pred>0))
    smape = 100*np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-6))
    evs = explained_variance_score(y_true, y_pred)
    return {"MSE":mse,"RMSE":rmse,"MAE":mae,"R2":r2,"MAPE":mape,"DA":da,"SMAPE":smape,"EVS":evs}

# ---------------- DL MODELS ----------------
def build_lstm(input_shape, units=16, dropout=0.3, H=4):
    model = Sequential([LSTM(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape, units=16, dropout=0.3, H=4):
    model = Sequential([GRU(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_conv1d(input_shape, units=16, dropout=0.3, kernel_size=3, H=4):
    model = Sequential([Conv1D(filters=units, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
                        Flatten(),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------- WALK-FORWARD ML ----------------
def walk_forward_ml(model, X, y, W=25, H=4, step=4):
    X_flat = X.reshape((X.shape[0], -1))
    metrics_list = []
    for start in range(0, len(X_flat)-W-H, step):
        train_end = start + W
        test_end = train_end + H
        if test_end > len(X_flat): break
        X_train, y_train = X_flat[:train_end], y[:train_end]
        X_test, y_test = X_flat[train_end:test_end], y[train_end:test_end]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_fit = MultiOutputRegressor(model) if y.shape[1]>1 else model
        model_fit.fit(X_train_scaled, y_train)
        y_pred = model_fit.predict(X_test_scaled)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

# ---------------- WALK-FORWARD DL ----------------
def walk_forward_dl(model_builder, X, y, W=25, H=4, step=4, units=16, dropout=0.3, epochs=10, batch_size=32, kernel_size=3):
    X_seq = X.reshape((X.shape[0], X.shape[1],1))
    metrics_list=[]
    for start in range(0, len(X_seq)-W-H, step):
        train_end=start+W
        test_end=train_end+H
        if test_end>len(X_seq): break
        X_train, y_train = X_seq[:train_end], y[:train_end]
        X_test, y_test = X_seq[train_end:test_end], y[train_end:test_end]
        # Escalado
        scaler = StandardScaler()
        X_train_2d = X_train.reshape((X_train.shape[0], -1))
        X_test_2d = X_test.reshape((X_test.shape[0], -1))
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)
        # Construir modelo
        model = model_builder(input_shape=(X_train.shape[1],1), units=units, dropout=dropout, H=H, kernel_size=kernel_size)
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_test_scaled, y_test), verbose=0, callbacks=[es, lr])
        y_pred = model.predict(X_test_scaled, verbose=0)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

# ---------------- MAIN PIPELINE ----------------
if __name__=="__main__":
    path_csv = "stock_details_5_years.csv"
    W,H = 25,4
    sub_sample_ratio = 0.3
    np.random.seed(42)
    
    df_weekly = load_and_pivot(path_csv)
    sampled_cols = np.random.choice(df_weekly.columns, int(len(df_weekly.columns)*sub_sample_ratio), replace=False)
    df_sub = df_weekly[sampled_cols]
    
    X_sub, y_sub = create_supervised(df_sub, W=W, H=H)
    X_sub = np.nan_to_num(X_sub)
    y_sub = np.nan_to_num(y_sub)
    
    # ML hyperparameters
    xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
    lgb_params = {"n_estimators":[100,200], "max_depth":[5,10], "learning_rate":[0.01,0.05]}
    
    # DL hyperparameters
    dl_params = {'units':[16,32], 'dropout':[0.2,0.3]}
    c1d_params = {'units':[16,32], 'dropout':[0.2,0.3], 'kernel_size':[3,5]}
    
    best_models={}
    results_sub={}
    tuning_results={}
    
    # ---- XGB tuning ----
    best_score, best_model = -np.inf, None
    for params in ParameterGrid(xgb_params):
        print(f"Probando XGB con {params}")
        model = XGBRegressor(random_state=42, verbosity=0, **params)
        metrics = walk_forward_ml(model, X_sub, y_sub, W=W, H=H, step=4)
        tuning_results.setdefault('XGB', []).append({'params': params, 'metrics': metrics})
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = model
    results_sub['XGB'] = walk_forward_ml(best_model, X_sub, y_sub, W=W,H=H,step=4)
    best_models['XGB'] = best_model
    
    # ---- LGBM tuning ----
    best_score, best_model = -np.inf, None
    for params in ParameterGrid(lgb_params):
        print(f"Probando LGBM con {params}")
        model = LGBMRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                              learning_rate=params['learning_rate'], random_state=42)
        metrics = walk_forward_ml(model, X_sub, y_sub, W=W,H=H,step=4)
        tuning_results.setdefault('LGBM', []).append({'params': params, 'metrics': metrics})
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = model
    results_sub['LGBM'] = walk_forward_ml(best_model, X_sub, y_sub, W=W,H=H,step=4)
    best_models['LGBM'] = best_model
    
    # ---- LSTM tuning ----
    best_score, best_config = -np.inf, None
    for units, dropout in product(dl_params['units'], dl_params['dropout']):
        print(f"Probando LSTM con {params}")
        metrics = walk_forward_dl(build_lstm, X_sub, y_sub, W=W,H=H,step=4,units=units,dropout=dropout,epochs=10,batch_size=32)
        config = {'units': units, 'dropout': dropout}
        tuning_results.setdefault('LSTM', []).append({'params': config, 'metrics': metrics})
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_config = config
    results_sub['LSTM'] = {'metrics': best_score, 'config': best_config}
    best_models['LSTM'] = best_config
    
    # ---- GRU tuning ----
    best_score, best_config = -np.inf, None
    for units, dropout in product(dl_params['units'], dl_params['dropout']):
        print(f"Probando GRU con {params}")
        metrics = walk_forward_dl(build_gru, X_sub, y_sub, W=W,H=H,step=4,units=units,dropout=dropout,epochs=10,batch_size=32)
        config = {'units': units, 'dropout': dropout}
        tuning_results.setdefault('GRU', []).append({'params': config, 'metrics': metrics})
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_config = config
    results_sub['GRU'] = {'metrics': best_score, 'config': best_config}
    best_models['GRU'] = best_config
    
    # ---- Conv1D tuning ----
    best_score, best_config = -np.inf, None
    for units, dropout, kernel_size in product(c1d_params['units'], c1d_params['dropout'], c1d_params['kernel_size']):
        print(f"Probando Conv1D con {params}")
        metrics = walk_forward_dl(build_conv1d, X_sub, y_sub, W=W,H=H,step=4,units=units,dropout=dropout,epochs=10,batch_size=32,kernel_size=kernel_size)
        config = {'units': units, 'dropout': dropout, 'kernel_size': kernel_size}
        tuning_results.setdefault('Conv1D', []).append({'params': config, 'metrics': metrics})
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_config = config
    results_sub['Conv1D'] = {'metrics': best_score, 'config': best_config}
    best_models['Conv1D'] = best_config
    
    print("=== Mejores modelos sobre submuestra ===")
    for k,v in results_sub.items(): print(k,v)
    
    """# ------------------- Entrenar con todas las empresas -------------------
    X_full,y_full=create_supervised(df_weekly,W=W,H=H)
    X_full=np.nan_to_num(X_full); y_full=np.nan_to_num(y_full)
    
    # Escalador para ML
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    
    final_models={}
    # ML: reentrenar
    for ml in ['XGB','LGBM']:
        model = best_models[ml]
        model_fit = MultiOutputRegressor(model) if y_full.shape[1]>1 else model
        model_fit.fit(X_full_scaled, y_full)
        final_models[ml]=model_fit
    
    # DL: reentrenar
    for dl_name, builder in zip(['LSTM','GRU','Conv1D'],[build_lstm,build_gru,build_conv1d]):
        config = best_models[dl_name]
        if dl_name=='Conv1D':
            model = builder(input_shape=(X_full.shape[1],1), units=config['units'],
                            dropout=config['dropout'], kernel_size=config['kernel_size'], H=H)
        else:
            model = builder(input_shape=(X_full.shape[1],1), units=config['units'], dropout=config['dropout'], H=H)
        X_full_seq = X_full_scaled.reshape((X_full.shape[0], X_full.shape[1],1))
        es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)
        model.fit(X_full_seq, y_full, epochs=20, batch_size=32, verbose=0, callbacks=[es,lr])
        final_models[dl_name]=model
    
    print("=== Modelos finales entrenados con todas las empresas ===")"""
