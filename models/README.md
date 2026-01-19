# Models

This folder contains all **model definitions** used in this project : 

- Centralize all forecasting models
- Ensure clean separation between **model definition** and **model execution**
- Enable fair benchmarking across different model families

No model is trained or executed in this folder.

## 📦 Model Families Included

### 1️⃣ Classical Machine Learning Models
- LightGBM
- XGBoost
- Random Forest

These models are used as strong tabular baselines for demand forecasting
with engineered lag, rolling, price and promotion features.

### 2️⃣ Deep Learning Models
- LSTM-based sequence models

These models are designed to capture long-term temporal dependencies
and seasonality patterns.

### 3️⃣ Clustering-based Models (Hybrid Approach)
- SKU clustering using historical demand patterns
- Cluster-specific ML models

This hybrid approach reduces heterogeneity across SKUs by learning
specialized models per demand regime (e.g. smooth, intermittent, seasonal).

## 🧠 Design Philosophy

- **models/** → model architecture & logic only
- **notebooks/** → orchestration, training, evaluation
- **results/** → metrics, comparison and interpretation

