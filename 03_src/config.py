import polars as pl
from pathlib import Path

# ── Pfade ────────────────────────────────────────────────────────────────────
RAW       = Path('..') / '02_data' / 'raw'
PROCESSED = Path('..') / '02_data' / 'processed'
FINAL     = Path('..') / '02_data' / 'final'
RESULTS = FINAL / 'results'

# ── Spalten ──────────────────────────────────────────────────────────────────
TARGET_COL = 'transactions'

CAT_COLS = ['hol_type_national']

# Alle numerischen Feature-Spalten (kein date, store_nbr, target)
FEATURE_COLS = [
    'store_type_enc', 'cluster',
    'month', 'weekday', 'quarter',
    'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 'year',
    'lag_1', 'lag_7', 'lag_14', 'lag_21', 'lag_28',
    'rolling_mean_7_days', 'rolling_mean_14_days', 'rolling_mean_28_days',
    'wow_growth', 'mom_growth',
    'same_weekday_last_week', 'same_weekday_4weeks_ago',
    'sales_vs_week_avg', 'wow_vs_month_avg',
    'diff_1', 'diff_7', 'diff_28',
    'oil_price', 'oil_price_ma7',
    'is_national_holiday', 'is_day_before_holiday',
    'is_day_after_holiday', 'is_holiday_window',
]

STAT_EXOG = ['store_type_enc', 'cluster']
HIST_EXOG = ['oil_price', 'is_national_holiday', 'is_day_before_holiday', 'is_day_after_holiday']
EXOG_COLS = ['oil_price', 'is_national_holiday', 'is_day_before_holiday', 'is_day_after_holiday']

TRAIN_END = pl.date(2016, 12, 31)   # Train: 2013-01-29 → 2016-12-31
VAL_END   = pl.date(2017,  5, 31)   # Val:   2017-01-01 → 2017-05-31
                                     # Test:  2017-06-01 → 2017-08-15

LOOKBACK = 336   # Kontextfenster in Tagen (~1 Jahr Wochenmuster)
HORIZON  =  30   # Forecast-Horizont in Tagen


MODELL_ORDER  = ['SARIMAX', 'Prophet', 'XGBoost', 'LightGBM', 'PatchTST', 'NHITS']
FILE_NAMES = {
    'SARIMAX':  'sarimax',
    'Prophet':  'prophet',
    'XGBoost':  'xgb',    
    'LightGBM': 'lgbm',      
    'PatchTST': 'patchtst',
    'NHITS':    'nhits',
}
MODELL_COLORS = {
    'SARIMAX':  'royalblue',
    'Prophet':  'darkorange',
    'XGBoost':  'seagreen',
    'LightGBM': 'purple',
    'PatchTST': 'tomato',
    'NHITS':    'sienna',
}
PRED_COLS = {
    'SARIMAX':  'pred_sarimax',
    'Prophet':  'pred_prophet',
    'XGBoost':  'pred_xgb',
    'LightGBM': 'pred_lgbm',
    'PatchTST': 'pred_patchtst',
    'NHITS':    'pred_nhits',
}

