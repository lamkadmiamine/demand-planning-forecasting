# 1. Data Exploration & Feature Engineering

This notebook focuses on exploring demand patterns at SKU level and preparing
a clean and structured dataset for multi-SKU demand forecasting.

The analysis is conducted on a subset of the M5 dataset and serves as the
foundation for machine learning and deep learning models developed later.

## 1.1 Exploratory Data Analysis (EDA)

This section explores the main characteristics of the demand data:
- Distribution of demand values
- Frequency of zero-demand periods
- Seasonality and temporal patterns across SKUs

![EDA1](https://github.com/lamkadmiamine/demand-planning-forecasting/blob/main/assets/Croissance_Annuelle_par_Categorie.png)

The analysis of year-over-year sales growth reveals distinct behaviors across
product categories.  
Household and Hobbies categories exhibit a clear and sustained upward trend,
indicating structurally growing demand.

In contrast, the Food category shows only marginal growth over the observed period
and even records a decline in 2015 compared to 2014 (-4.1%). This suggests a more
mature or saturated demand pattern, with higher sensitivity to external factors
such as price and promotions.

These differences justify the inclusion of category-level promotional indicators
(e.g. `promo_foods`, `promo_hobbies`, `promo_household`) to allow models to capture
heterogeneous demand dynamics across categories.

<p align="center">
  <img src="https://github.com/lamkadmiamine/demand-planning-forecasting/blob/main/assets/Ventes_Moyenne_avecETsans_Promo.png" width="45%" />
  <img src="https://github.com/lamkadmiamine/demand-planning-forecasting/blob/main/assets/Promo_Impact_boxplot.png" width="45%" />
</p>


![EDA2](https://github.com/lamkadmiamine/demand-planning-forecasting/blob/main/assets/Ventes_Moyenne_avecETsans_Promo.png)

A comparison of average sales with and without promotions highlights the strong
effect of promotional activities on demand.

Across all categories, average sales during promotional periods are significantly
higher than during non-promotional periods. This confirms that promotions act as a
major demand accelerator and must be explicitly modeled to avoid biased forecasts.

As a result, multiple promotion-related features are engineered, including:
- Promotion flags
- Recent promotion history (e.g. last 3 periods)
- Category-specific promotion indicators

These features enable machine learning and deep learning models to better capture
demand uplift effects driven by promotional campaigns.

## 1.2 Feature Engineering

Feature engineering aims to enrich the dataset with explanatory variables that
capture demand dynamics, seasonality, price effects and promotional impacts.

![FE1](https://github.com/lamkadmiamine/demand-planning-forecasting/blob/main/assets/Promo_Impact_boxplot.png)

### 1.2.1 Engineered Feature Categories

The following feature groups are created:

- **Zero-demand indicators**
- **Time-based features**
- **Lagged demand features**
- **Rolling statistics**
- **Price-related features**
- **Promotion-related features**
