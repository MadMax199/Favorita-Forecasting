import os
import pathlib
import tempfile

os.environ['CMDSTANPY_NOT_USE_POLARS'] = 'True'  # ✅ muss VOR prophet-Import stehen

import polars as pl
import numpy as np
# import pathlib  ← Duplikat, weg damit
import pandas as pd

from config import TARGET_COL, EXOG_COLS, HIST_EXOG, STAT_EXOG, RESULTS
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm


# ── Hilfsfunktion: Metriken ──────────────────────────────────────────────────

def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    if label:
        print(f"{label:20s}  MAE={mae:.1f}  RMSE={rmse:.1f}  MAPE={mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}



# ── 1 · SARIMAX (KORRIGIERTE VERSION) ──────────────────────────────────────────
def run_sarimax(train_df, val_df, stores, exog_cols=EXOG_COLS):
    all_preds = []

    for store_id in tqdm(stores, desc="SARIMAX"):
        # 1. Daten filtern und zu Pandas
        s_train_pl = train_df.filter(pl.col('store_nbr') == store_id).sort('date')
        s_val_pl = val_df.filter(pl.col('store_nbr') == store_id).sort('date')

        # 2. Datentypen prüfen: Nur numerische exogene Spalten verwenden
        valid_exog = [c for c in exog_cols if train_df.schema[c] in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Boolean]]
        
        s_train = s_train_pl.to_pandas().set_index('date')
        s_val = s_val_pl.to_pandas().set_index('date')

        try:
            # 3. Vorbereitung der Daten für statsmodels
            y_train = s_train[TARGET_COL].astype(float)
            X_train = s_train[valid_exog].astype(float)
            X_val = s_val[valid_exog].astype(float)

            # 4. Modell-Definition und Fit
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)

            # 5. Forecast erstellen
            forecast = fit.get_forecast(
                steps=len(s_val),
                exog=X_val
            )

            # 6. Ergebnisse sammeln und Datentyp-Fix anwenden
            all_preds.append(pl.DataFrame({
                # FIX: Index explizit als String-Liste exportieren, damit Polars kein 'Object' erzeugt
                'date': s_val.index.astype(str).tolist(), 
                'store_nbr': [store_id] * len(s_val),
                'y_true': s_val[TARGET_COL].values,
                'pred_sarimax': forecast.predicted_mean.values,
            }))

        except Exception as e:
            print(f"  Fehler Store {store_id}: {e}")

    # Zusammenfügen und das Datum final in einen echten Polars-Date-Typ umwandeln
    result_df = pl.concat(all_preds)
    
    return result_df.with_columns(
        pl.col("date").str.to_date()
    )

# ── 3 · Prophet ──────────────────────────────────────────────────────────────

def run_prophet(train_df, val_df, stores, extra_regressors=None):
    if extra_regressors is None:
        extra_regressors = ['oil_price']

    all_preds = []

    for store_id in tqdm(stores, desc="Prophet Progress"):
        train_s = train_df.filter(pl.col('store_nbr') == store_id).sort('date')
        val_s   = val_df.filter(pl.col('store_nbr') == store_id).sort('date')

        if train_s.is_empty():
            continue

        df_train = train_s.select(
            [pl.col('date').alias('ds'), pl.col('transactions').alias('y')] +
            [pl.col(c) for c in extra_regressors]
        ).to_pandas()
        df_train['ds'] = pd.to_datetime(df_train['ds'])

        df_val = val_s.select(
            [pl.col('date').alias('ds')] +
            [pl.col(c) for c in extra_regressors]
        ).to_pandas()
        df_val['ds'] = pd.to_datetime(df_val['ds'])

        if df_train['y'].isna().any() or (df_train['y'] == 0).all():
            print(f"Store {store_id}: übersprungen (NaN oder nur Nullen)")
            continue

        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=1.0,
            )
            for reg in extra_regressors:
                model.add_regressor(reg)

            model.fit(df_train)
            forecast = model.predict(df_val)

            all_preds.append(pl.DataFrame({
                'date':         val_s['date'].cast(pl.String).to_list(),
                'store_nbr':    [store_id] * len(val_s),
                'y_true':       val_s['transactions'].cast(pl.Float64).to_list(),
                'pred_prophet': forecast['yhat'].values.astype(float),
            }))

        except Exception as e:
            print(f"Store {store_id}: fehlgeschlagen – {str(e)[:150]}")
            continue

    if not all_preds:
        print("KRITISCH: Keine Vorhersagen erstellt!")
        return pl.DataFrame()

    return pl.concat(all_preds).with_columns(pl.col("date").str.to_date())

# ── 3 · XGBoost ──────────────────────────────────────────────────────────────

def run_xgb(X_train, y_train, X_val, y_val):
    """
    Trainiert XGBoost und gibt (model, mae_val) zurück.
    """
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_val)
    return model, evaluate(y_val, preds, label="XGBoost Val")


# ── 4 · LightGBM ─────────────────────────────────────────────────────────────

def run_lgbm(X_train, y_train, X_val, y_val):
    """
    Trainiert LightGBM und gibt (model, mae_val) zurück.
    """
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )
    preds = model.predict(X_val)
    return model, evaluate(y_val, preds, label="LightGBM Val")

# ── 4 · NF_format ─────────────────────────────────────────────────────────────


def to_nf_format(df):
    return (
        df
        .select([
            pl.col('store_nbr').cast(pl.Utf8).alias('unique_id'),
            pl.col('date').alias('ds'),
            pl.col(TARGET_COL).alias('y'),
            pl.col('store_type_enc').cast(pl.Float32),
            pl.col('cluster').cast(pl.Float32),
            pl.col('oil_price').cast(pl.Float32),
            pl.col('is_national_holiday').cast(pl.Float32),
            pl.col('is_day_before_holiday').cast(pl.Float32),
            pl.col('is_day_after_holiday').cast(pl.Float32),
        ])
        .to_pandas()
    )

# ── 5 · Evaluation Part ─────────────────────────────────────────────────────────────



def compute_metrics(y_true, y_pred, label, split):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {'modell': label, 'split': split, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def load_preds(fname):
    path = RESULTS / fname
    if path.exists():
        return pl.read_parquet(path)
    print(f'⚠  {fname} nicht gefunden — wird übersprungen')
    return None