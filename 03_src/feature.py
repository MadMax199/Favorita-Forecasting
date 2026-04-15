
import polars as pl
import numpy as np


def add_lag(df, lags):
    """
    Erstellt Lag Features für die Sales Spalten
    """
    return df.with_columns([
        pl.col("sales").shift(lag).over("platform").alias(f"lag_{lag}")
        for lag in lags
    ])

def add_rolling_mean(df, window_sizes, group_col="platform", target_col="sales"):
    """
    Erstellt Lag Features für die Sales Spalten
    """
    expressions = [
        pl.col(target_col)
        .rolling_mean(window_size=w)
        .over(group_col)
        .alias(f"rolling_mean_{w}_months")
        for w in window_sizes
    ]
    
    # Alle Spalten in einem Rutsch hinzufügen
    return df.with_columns(expressions)

def add_momentum_features(df, over="platform", target_col="sales"):
    """
    Berechnet Wachstumsraten (Momentum) für monatliche Daten.
    - mom_growth: Vergleich zum Vormonat
    - yoy_growth: Vergleich zum Vorjahresmonat (12 Monate Lag)
    """
    return df.with_columns([
        ((pl.col(target_col) / pl.col(target_col).shift(1).over(over)) - 1)
        .alias("mom_growth"),        
        ((pl.col(target_col) / pl.col(target_col).shift(12).over(over)) - 1)
        .alias("yoy_growth")
    ])

def add_cyclical_month_features(df):
    return df.with_columns([
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.quarter().alias("quarter"),
        (pl.col("datetime").dt.month() * (2 * np.pi / 12)).sin().alias("month_sin"),
        (pl.col("datetime").dt.month() * (2 * np.pi / 12)).cos().alias("month_cos")
    ])

def add_historical_benchmarks(df, over_cols=["platform"], target_col="sales"):
    """
    Erstellt Benchmarks basierend auf dem Vorjahr.
    Da Daten monatlich sind, ist der 'Vorjahresmonat' der sauberste Vergleich.
    """
    return df.with_columns([
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").dt.month().alias("month"),
    
        pl.col(target_col)
        .shift(12)
        .over(over_cols)
        .alias("avg_sales_last_year_same_month")
    ])

def add_weekend_features(df, date_col="datetime"):
    """
    Markiert, ob das Datum (Monatsende) auf ein Wochenende fällt.
    Polars dt.weekday() liefert: 1 (Mo) bis 7 (So).
    """
    return df.with_columns([
        pl.col(date_col).dt.strftime("%A").alias("day_name"),
        pl.col(date_col).dt.weekday().is_in([6, 7]).cast(pl.Int8).alias("is_weekend")
    ])

def add_ratio_features(df):
    return df.with_columns([
        (pl.col("sales") / pl.col("rolling_mean_3_months"))
        .alias("sales_vs_recent_avg"),

        (pl.col("lag_12") / pl.col("rolling_mean_12_months"))
        .alias("seasonality_strength"),
    ])


def add_difference_features(df):
    return df.with_columns([
        (pl.col("sales") - pl.col("sales").shift(1).over("platform"))
        .alias("diff_1"),

        (pl.col("sales") - pl.col("sales").shift(12).over("platform"))
        .alias("diff_12"),
    ])