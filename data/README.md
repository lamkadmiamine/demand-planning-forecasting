# Data Description

## Raw Data

The initial dataset is based on a processed version of the M5 forecasting dataset (`m5_processed_complete.parquet`), containing historical sales, prices and promotional information across multiple stores and product categories.

Due to memory and RAM constraints, the analysis and modeling were performed on a single store (`CA_1`).
This choice allows focusing on the full modeling pipeline while preserving the multi-SKU forecasting
complexity within a realistic industrial constraint.

## Data Preparation

Data preparation and feature engineering are performed in the data exploration notebook. The final modeling dataset is saved as `df_model_final.parquet`.

The preprocessing steps include:
- Filtering data for store `CA_1`
- Handling zero-demand periods
- Creation of time-based features
- Engineering lagged and rolling statistics
- Integration of price and promotion-related features

## Final Feature Set

The final dataset used for modeling contains the following features:

### Target-related indicators
- `is_zero`
- `was_zero_last_month`

### Time features
- `month_sin`, `month_cos`
- `time_idx`

### Promotion features
- `promo_flag`
- `promo_last_3`
- `promo_foods`
- `promo_hobbies`
- `promo_household`

### Price features
- `sell_price`
- `price_lag_1`
- `price_change_1`

### Lagged demand features
- `lag_1, lag_2, ..., lag_n`

### Rolling statistics
- `roll_mean_3`
- `roll_mean_6`
- `roll_std_6`

This structured feature set enables machine learning and deep learning models to capture seasonality, trends, price sensitivity and promotional effects across multiple SKUs.

