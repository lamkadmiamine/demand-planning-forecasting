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

## 1.2 Feature Engineering

Feature engineering aims to enrich the dataset with explanatory variables that
capture demand dynamics, seasonality, price effects and promotional impacts.

### 1.2.1 Engineered Feature Categories

The following feature groups are created:

- **Zero-demand indicators**
- **Time-based features**
- **Lagged demand features**
- **Rolling statistics**
- **Price-related features**
- **Promotion-related features**
