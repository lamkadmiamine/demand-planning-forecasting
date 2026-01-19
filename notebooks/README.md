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
  <img src="https://github.com/lamkadmiamine/demand-planning-forecasting/blob/main/assets/Ventes_Moyenne_avecETsans_Promo.png" height="280" />
  <img src="https://github.com/lamkadmiamine/demand-planning-forecasting/blob/main/assets/Promo_Impact_boxplot.png" height="280" />
</p>


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

## Diagnostic Summary (Feature Insights)

| **Question / Test** | **Method** | **Result** | **Interpretation (Implication for Forecasting)** |
|---|---|---|---|
| **Do promotions explain sales?** | Mann-Whitney U test (log sales) + boxplot | **p-value < 0.05** (significant) | Promotion is a strong demand driver. Include `promo_flag` as a key feature. |
| **Does price significantly impact sales?** | Spearman correlation (non-linear) | Moderate negative correlation | Price is informative. Use `sell_price` as a feature, potentially with non-linear transformations. |
| **Is the effect linear or not?** | Spearman correlation (non-linear) | Non-linear pattern observed | Consider non-linear models (e.g., tree-based) or polynomial/interaction terms. |
| **Is the effect different by category/department?** | Spearman correlation by `cat_id` | Variation across categories | Build category-specific models or include interactions (category × price/discount). |
| **Does discount explain sales?** | Spearman correlation with `discount_pct` | Weak/negative correlation | Discount alone is not a stable driver; it is contextual. |
| **Do exogenous variables explain the probability of selling?** | Logistic Regression (sold vs not sold) | Promo and price are significant predictors | Use a classification step or a hurdle model for zero-inflated sales. |
| **What explains sales best?** | Combined insights | Promo + price + seasonality + category | Build forecasting features: `promo_flag`, `sell_price`, `seasonality`, `cat_id`, `dept_id`. |
| **What is contextual?** | Observation from data | Discount appears when demand is low | `discount_pct` should be treated as **contextual** (not causal) and used cautiously. |


The following feature groups are created:

- **Zero-demand indicators**
- **Time-based features**
- **Lagged demand features**
- **Rolling statistics**
- **Price-related features**
- **Promotion-related features**


# 2. SKUs Forecasting

This file contains the **orchestration layer** of the project:

- Execute the full forecasting pipeline
- Run experiments in a reproducible way
- Compare multiple forecasting approaches on a common test set

All **model definitions** are implemented in the `models/` folder.
No modeling logic is duplicated here.

- Loads the prepared monthly dataset
- Applies strict temporal train / validation / test splits
- Trains multiple model families:
  - Classical ML models
  - Cluster-based hybrid models
  - Deep Learning (LSTM)
- Evaluates all models using consistent metrics
- Produces a unified benchmark table

The notebook acts as the **entry point** for model comparison.

