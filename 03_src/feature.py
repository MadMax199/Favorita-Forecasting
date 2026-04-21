import polars as pl
import numpy as np


# ──────────────────────────────────────────────
# Kern-Features (aus altem Projekt übernommen,
# Parameter auf tägliche Daten angepasst)
# ──────────────────────────────────────────────

def add_lag(df, lags, group_col="store_nbr", target_col="transactions"):
    """
    Lag-Features für die Zielspalte.

    Relevante Lags (täglich, Wochenmultiplikatoren dominieren):
        7, 14, 21, 28  → ACF > 0.68
        1              → kurzfristige Abhängigkeit
    """
    return df.with_columns([
        pl.col(target_col)
        .shift(lag)
        .over(group_col)
        .alias(f"lag_{lag}")
        for lag in lags
    ])


def add_rolling_mean(df, window_sizes, group_col="store_nbr", target_col="transactions"):
    """
    Rolling-Mean-Features.

    Sinnvolle Window-Größen für tägliche Daten:
        7  → Wochenmittel
        14 → Zweiwochenmittel
        28 → Monatsmittel
    """
    return df.with_columns([
        pl.col(target_col)
        .rolling_mean(window_size=w)
        .over(group_col)
        .alias(f"rolling_mean_{w}_days")
        for w in window_sizes
    ])


def add_momentum_features(df, over="store_nbr", target_col="transactions"):
    """
    Wachstumsraten für tägliche Daten.
    - wow_growth: Vergleich zur Vorwoche (7 Tage)
    - mom_growth: Vergleich zum Vormonat (28 Tage)

    Hinweis: yoy_growth (365 Tage) ist bei Favorita nicht sinnvoll,
    da ACF Lag 365 ≈ -0.23 (keine Jahressaisonalität).
    """
    return df.with_columns([
        ((pl.col(target_col) / pl.col(target_col).shift(7).over(over)) - 1)
        .alias("wow_growth"),

        ((pl.col(target_col) / pl.col(target_col).shift(28).over(over)) - 1)
        .alias("mom_growth"),
    ])


def add_cyclical_time_features(df):
    """
    Zyklische Encoding für Wochentag und Monat.
    Neu gegenüber altem Projekt: Wochentag (statt nur Monat),
    da tägliche Daten eine starke Wochensaisonalität zeigen (ACF 7 ≈ 0.76).
    """
    return df.with_columns([
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("weekday"),      # 0=Mo, 6=So
        pl.col("date").dt.quarter().alias("quarter"),

        # Wochentag zyklisch
        (pl.col("date").dt.weekday() * (2 * np.pi / 7)).sin().alias("weekday_sin"),
        (pl.col("date").dt.weekday() * (2 * np.pi / 7)).cos().alias("weekday_cos"),

        # Monat zyklisch
        (pl.col("date").dt.month() * (2 * np.pi / 12)).sin().alias("month_sin"),
        (pl.col("date").dt.month() * (2 * np.pi / 12)).cos().alias("month_cos"),
    ])


def add_historical_benchmarks(df, over_cols=["store_nbr"], target_col="transactions"):
    """
    Benchmark: Gleicher Wochentag der Vorwoche (Lag 7).
    Ersetzt den Vorjahresmonat aus dem alten Projekt —
    bei täglichen Daten ist der 7-Tage-Benchmark der sauberste Vergleich.
    """
    return df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),

        pl.col(target_col)
        .shift(7)
        .over(over_cols)
        .alias("same_weekday_last_week"),

        pl.col(target_col)
        .shift(28)
        .over(over_cols)
        .alias("same_weekday_4weeks_ago"),
    ])


def add_ratio_features(df):
    """
    Verhältnis-Features: aktueller Wert vs. gleitende Mittel.
    Window-Namen angepasst auf tägliche Granularität (_days statt _months).
    """
    return df.with_columns([
        (pl.col("transactions") / pl.col("rolling_mean_7_days"))
        .alias("sales_vs_week_avg"),

        (pl.col("lag_7") / pl.col("rolling_mean_28_days"))
        .alias("wow_vs_month_avg"),
    ])


def add_difference_features(df):
    """
    Differenz-Features.
    Lag 7 statt Lag 1/12 aus dem alten Projekt,
    da Lag 7 die stärkste Autokorrelation hat.
    """
    return df.with_columns([
        (pl.col("transactions") - pl.col("transactions").shift(1).over("store_nbr"))
        .alias("diff_1"),

        (pl.col("transactions") - pl.col("transactions").shift(7).over("store_nbr"))
        .alias("diff_7"),

        (pl.col("transactions") - pl.col("transactions").shift(28).over("store_nbr"))
        .alias("diff_28"),
    ])


# ──────────────────────────────────────────────
# Neue Features (Favorita-spezifisch)
# ──────────────────────────────────────────────

def add_oil_features(df, oil_df):
    """
    Ölpreis als exogener Regressor.
    - Join auf date, dann forward_fill auf dem vollständigen Datumsbereich.
      (Vorher füllen reicht nicht: Tage die nur in tx, nicht in oil_df existieren
      — z.B. Wochenenden — bekommen sonst nach dem Join ein NaN das nicht gefüllt wird.)
    - Zusätzlich: 7-Tage Rolling Mean des Ölpreises (glättet Rauschen).

    oil_df: Polars DataFrame mit Spalten ['date', 'dcoilwtico']
    """
    oil_clean = oil_df.sort("date").select(["date", "dcoilwtico"])

    return (
        df
        .join(oil_clean, on="date", how="left")
        .sort(["store_nbr", "date"])
        .with_columns(
            pl.col("dcoilwtico")
            .forward_fill()
            .over("store_nbr")
            .alias("oil_price")
        )
        .drop("dcoilwtico")
        .with_columns(
            pl.col("oil_price")
            .rolling_mean(window_size=7)
            .over("store_nbr")
            .alias("oil_price_ma7")
        )
    )


def add_holiday_features(df, hol_df):
    """
    Feiertags-Features aus holidays_events.csv.

    Drei Feature-Gruppen:
    1. is_national_holiday   → nationaler Feiertag (alle Stores betroffen)
    2. is_holiday_window     → Tag vor oder nach einem nat. Feiertag
       (EDA zeigt: +13% am Tag vor, +11% am Tag nach Feiertagen)
    3. hol_type_national     → Typ des nationalen Feiertags (für Modell als Kategorie)

    Transferred=True Einträge werden herausgefiltert —
    diese Tage sind *keine* Feiertage mehr (der Feiertag wurde verschoben).
    """
    # Nur nationale, nicht-transferierte Feiertage
    nat_hol = (
        hol_df
        .filter(
            (pl.col("locale") == "National") &
            (pl.col("transferred") == False)
        )
        .select(["date", "type"])
        .rename({"type": "hol_type_national"})
    )

    df = df.join(nat_hol, on="date", how="left")

    df = df.with_columns([
        # Binäres Holiday-Flag
        pl.col("hol_type_national").is_not_null().alias("is_national_holiday"),

        # Typ als String (null = kein Feiertag)
        pl.col("hol_type_national").fill_null("none"),
    ])

    # Holiday-Window: Tag davor / danach
    hol_dates = nat_hol.select("date")

    df = df.with_columns([
        # Tag vor einem Feiertag: date + 1 Tag ist ein Feiertag
        (pl.col("date") + pl.duration(days=1))
        .is_in(hol_dates.to_series())
        .alias("is_day_before_holiday"),

        # Tag nach einem Feiertag
        (pl.col("date") - pl.duration(days=1))
        .is_in(hol_dates.to_series())
        .alias("is_day_after_holiday"),
    ])

    # Kombiniertes Window-Flag
    df = df.with_columns(
        (pl.col("is_day_before_holiday") | pl.col("is_day_after_holiday"))
        .alias("is_holiday_window")
    )

    return df


def add_store_features(df, stores_df):
    """
    Store-Metadaten als Features.
    - store_type_enc: Ordinales Encoding (A=5 groß → E=1 klein)
    - cluster: direkt als numerisches Feature
    """
    type_map = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

    stores_feat = stores_df.with_columns(
        pl.col("type")
        .replace(type_map)
        .cast(pl.Int8)
        .alias("store_type_enc")
    ).select(["store_nbr", "store_type_enc", "cluster"])

    return df.join(stores_feat, on="store_nbr", how="left")
