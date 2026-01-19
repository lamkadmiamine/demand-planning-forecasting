## 📊 Benchmark Results

All models were evaluated on the same **strict temporal test set** using
multiple error metrics and execution time indicators.

| Model                     | MAE   | RMSE  | sMAPE | WAPE | Train Time (s) | Predict Time (s) |
|---------------------------|-------|-------|-------|-------|---------------|------------------|
| LightGBM                  | 11.10 | 28.34 | 50.63 | 0.269 | 2.70          | 0.13             |
| RandomForest              | 11.19 | 29.24 | 31.98 | 0.271 | 243.86        | 0.60             |
| RandomForest (Clustered)  | 11.20 | 28.96 | 32.18 | 0.272 | 266.91        | 0.74             |
| LSTM + LightGBM (Hybrid)  | 11.22 | 29.90 | 42.85 | 0.272 | 355.19        | 0.27             |
| LightGBM (Clustered)      | 11.24 | 28.04 | 50.91 | 0.272 | 9.15          | 0.50             |
| XGBoost                   | 11.30 | 28.84 | 50.73 | 0.274 | 7.23          | 0.30             |
| XGBoost (Clustered)       | 11.71 | 31.81 | 51.21 | 0.284 | 10.91         | 0.31             |
| LSTM                      | 26.97 | 71.58 | 77.49 | 0.654 | 483.22        | 1.64             |


Tree-based Machine Learning models clearly outperform Deep Learning approaches for this **multi-SKU monthly demand forecasting** task.
All classical ML models achieve similar error levels **(WAPE ≈ 0.27)**, while the LSTM model performs significantly worse due to high data sparsity and limited sequence length.
This indicates that **most of the predictive signal is already captured by feature engineering** (lags, rolling statistics, price and promotion effects).

**Classical ML (LightGBM, Random Forest, XGBoost)**

* Strong and stable performance across SKUs
* LightGBM achieves the best overall trade-off between accuracy and speed
* Random Forest is computationally expensive with no clear performance gain

**Cluster-based Models**

* Cluster-specific models perform similarly to global models
* No consistent improvement over global LightGBM
* Useful conceptually, but benefits depend on SKU heterogeneity and scale

**Deep Learning (LSTM)**

* LSTM underperforms due to sparse sales and long zero periods
* Hybrid LSTM + LightGBM does not bring clear gains
* High training and inference cost

### ⚠️ Limitations

* Monthly aggregation may hide short-term demand signals
* Cluster-based modeling increases complexity with limited performance gains

### 🚀 Future Improvements

* Hierarchical forecasting (bottom-up / top-down)
* Rolling-window backtesting
* Probabilistic forecasts (prediction intervals)
* Scenario simulation for pricing and promotions
* Advanced SKU clustering (shape-based, DTW)


