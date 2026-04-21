import polars as pl

TARGET_COL = "transactions"

FEATURE_COL = [
    "store_type_enc","cluster",	"month","weekday","quarter","weekday_sin","weekday_cos",
     "month_sin","month_cos","lag_1","lag_7","lag_14"	,"lag_21", "lag_28"	, "rolling_mean_7_days",
     "rolling_mean_14_days","rolling_mean_28_days","wow_growth","mom_growth", "year",
      "same_weekday_last_week","same_weekday_4weeks_ago","sales_vs_week_avg", "wow_vs_month_avg",	
      "diff_1", "diff_7", "diff_28","oil_price","oil_price_ma7", "hol_type_national",
      "is_national_holiday","is_day_before_holiday","is_day_after_holiday",	"is_holiday_window"
]

TRAIN_END = pl.date(2016, 12, 31)
VAL_END   = pl.date(2017,  5, 31)