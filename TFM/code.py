# -------------------------------
# Librerías
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras import layers, models, callbacks, optimizers

# -------------------------------
# Parámetros
# -------------------------------
DATA_PATH = 'stock_details_5_years.csv'
W = 30
H = 10
LR = 1e-3
EPOCHS = 20
BATCH_SIZE = 512
PATIENCE = 5
FEATURES = ['r1','ma4','ma12','vol4','vol12','mom_12_1']

# -------------------------------
# Cargar y preprocesar
# -------------------------------
def load_and_standardize(path=DATA_PATH):
    df = pd.read_csv(path)
    # Convertimos a datetime y normalizamos la zona horaria a UTC
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Company','Date']).reset_index(drop=True)
    
    # Agrupamos semanalmente por Company
    df_weekly = []
    for t, g in df.groupby('Company'):
        g = g.set_index('Date').resample('W').last().reset_index()
        df_weekly.append(g)
    
    df_final = pd.concat(df_weekly).reset_index(drop=True)
    return df_final


# -------------------------------
# EDA
# -------------------------------
def eda_report(df):
    print(df.head())
    print(df.describe())
    tickers = df['Company'].unique()[:5]
    plt.figure(figsize=(12,5))
    for t in tickers:
        g = df[df['Company']==t]
        plt.plot(g['Date'], g['Close'], label=t)
    plt.title("Precios de cierre (muestra)")
    plt.legend()
    plt.show()

# -------------------------------
# Ingeniería de features
# -------------------------------
def feature_factory(df):
    out = []
    for t, g in df.groupby('Company'):
        g = g.sort_values('Date').copy()
        g['log_close'] = np.log(g['Close'])
        g['r1'] = g['log_close'].diff()
        g['ma4'] = g['Close'].rolling(4).mean()
        g['ma12'] = g['Close'].rolling(12).mean()
        g['vol4'] = g['r1'].rolling(4).std()
        g['vol12'] = g['r1'].rolling(12).std()
        g['mom_12_1'] = g['Close'].pct_change(12) - g['Close'].pct_change(4)
        out.append(g)
    df_feat = pd.concat(out).dropna().reset_index(drop=True)
    return df_feat

# -------------------------------
# Análisis de correlación de features
# -------------------------------
def feature_correlation(df_feat, features=FEATURES):
    corr = df_feat[features].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlación entre features")
    plt.show()
    return corr

# -------------------------------
# Crear ventanas para series temporales
# -------------------------------
def make_windows(df_feat, W=W, H=H, features=FEATURES):
    tickers = df_feat['Company'].unique()
    ticker2id = {t:i for i,t in enumerate(tickers)}
    samples, ys, tids, dates = [],[],[],[]
    for t in tickers:
        g = df_feat[df_feat['Company']==t].sort_values('Date')
        arr = g[features].values
        logc = g['log_close'].values
        n = len(g)
        for i in range(W-1, n-H):
            samples.append(arr[i-W+1:i+1])
            ys.append(logc[i+H]-logc[i])
            tids.append(ticker2id[t])
            dates.append(g['Date'].iloc[i])
    return np.stack(samples), np.array(ys), np.array(tids), np.array(dates), ticker2id

# -------------------------------
# Modelos
# -------------------------------
def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(LR), loss='mse', metrics=['mae'])
    return model

def build_rnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(LR), loss='mse', metrics=['mae'])
    return model

def train_rf(X_train, y_train):
    ns, W_, nf = X_train.shape
    Xflat = X_train.reshape(ns, W_*nf)
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(Xflat, y_train)
    return rf

def train_xgb(X_train, y_train):
    ns, W_, nf = X_train.shape
    Xflat = X_train.reshape(ns, W_*nf)
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(Xflat, y_train)
    return xgb_model

def train_lgb(X_train, y_train):
    ns, W_, nf = X_train.shape
    Xflat = X_train.reshape(ns, W_*nf)
    lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    lgb_model.fit(Xflat, y_train)
    return lgb_model

# -------------------------------
# Métricas
# -------------------------------
def directional_accuracy(y_true, y_pred):
    return np.mean((y_true>0)==(y_pred>0))

def evaluate_model(model, X_test, y_test, is_flat=False):
    if is_flat:
        ns, W_, nf = X_test.shape
        Xflat = X_test.reshape(ns, W_*nf)
        y_pred = model.predict(Xflat)
    else:
        y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    da = directional_accuracy(y_test, y_pred)
    return mse, r2, da

# -------------------------------
# Main
# -------------------------------
df = load_and_standardize(DATA_PATH)
eda_report(df)
df_feat = feature_factory(df)
feature_correlation(df_feat, FEATURES)

X, y, tids, dates, ticker2id = make_windows(df_feat)

# Escalado
ns, W_, nf = X.shape
scaler = StandardScaler().fit(X.reshape(-1, nf))
Xs = scaler.transform(X.reshape(-1, nf)).reshape(X.shape)

# División train/test
split = int(len(Xs)*0.8)
X_train, X_test = Xs[:split], Xs[split:]
y_train, y_test = y[:split], y[split:]

results = {}

# LSTM
print("LSTM")
lstm = build_lstm((W_, nf))
es = callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
lstm.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
results['LSTM'] = evaluate_model(lstm, X_test, y_test)

# RNN
print("RNN")
rnn = build_rnn((W_, nf))
rnn.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
results['RNN'] = evaluate_model(rnn, X_test, y_test)

# XGBoost
print("XGBoost")
xgb_model = train_xgb(X_train, y_train)
results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, is_flat=True)

# LightGBM
print("LightGBM")
lgb_model = train_lgb(X_train, y_train)
results['LightGBM'] = evaluate_model(lgb_model, X_test, y_test, is_flat=True)

"""# Random Forest
print("Random Forest")
rf = train_rf(X_train, y_train)
results['Random Forest'] = evaluate_model(rf, X_test, y_test, is_flat=True)"""

# Comparación final
df_results = pd.DataFrame(results, index=['MSE','R2','Directional Accuracy']).T
print(df_results)
df_results.plot(kind='bar', figsize=(12,6), subplots=True, layout=(1,3), legend=False)
plt.suptitle("Comparación de modelos")
plt.tight_layout()
plt.show()

# -------------------------------
# Importancia de features
# -------------------------------
def feature_importance_rf(model, features):
    imp = pd.DataFrame({'Feature':features,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False)
    print("\nImportancia features RF:")
    print(imp)
    sns.barplot(x='Importance',y='Feature',data=imp)
    plt.title("RF Feature Importance")
    plt.show()
    return imp

def feature_importance_lgb(model, features):
    imp = pd.DataFrame({'Feature':features,'Importance':model.feature_importances_}).sort_values('Importance',ascending=False)
    print("\nImportancia features LGBM:")
    print(imp)
    sns.barplot(x='Importance',y='Feature',data=imp)
    plt.title("LGBM Feature Importance")
    plt.show()
    return imp

feature_importance_rf(xgb_model, FEATURES)
feature_importance_lgb(lgb_model, FEATURES)
