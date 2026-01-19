import pandas as pd

from models.ml_models import BenchmarkMLModels
from models.clustering_models import SKUClustering
from models.clustered_ml_models import ClusteredMLForecast
from models.lstm_models import LSTMModels
from models.base_metrics import ForecastMetrics


class ForecastingPipeline:
    """
    End-to-end orchestration of forecasting experiments.

    This class:
    - Applies temporal splitting
    - Trains multiple model families
    - Aggregates evaluation metrics
    """

    def __init__(self, df, feature_cols, target_col="y"):
        self.df = df.sort_values(
            ["item_id", "store_id", "year", "month"]
        ).reset_index(drop=True)

        self.feature_cols = feature_cols
        self.target_col = target_col

        self.results = []

    # ------------------------------------------------------------------
    # Temporal split (STRICT — no leakage)
    # ------------------------------------------------------------------
    @staticmethod
    def temporal_split(df, train_ratio=0.7, val_ratio=0.15):
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return (
            df.iloc[:train_end],
            df.iloc[train_end:val_end],
            df.iloc[val_end:]
        )

    # ------------------------------------------------------------------
    # 1️⃣ Classical ML models
    # ------------------------------------------------------------------
    def run_ml_models(self):
        df_train, df_val, df_test = self.temporal_split(self.df)

        ml = BenchmarkMLModels(
            feature_cols=self.feature_cols,
            target_col=self.target_col
        )

        self.results.append(
            ml.train_lightgbm(df_train, df_val, df_test)
        )

        self.results.append(
            ml.train_xgboost(df_train, df_val, df_test)
        )

        self.results.append(
            ml.train_random_forest(df_train, df_test)
        )

    # ------------------------------------------------------------------
    # 2️⃣ Clustering + ML (Hybrid)
    # ------------------------------------------------------------------
    def run_clustered_ml(self, n_clusters=4):
        clustering = SKUClustering(n_clusters=n_clusters)
        clustering.fit(self.df)

        df_clustered = clustering.assign_clusters(self.df)

        clustered_ml = ClusteredMLForecast(
            feature_cols=self.feature_cols,
            target_col=self.target_col
        )

        cluster_results = clustered_ml.train_per_cluster(df_clustered)

        # Agrégation globale pondérée par volume
        global_metrics = (
            cluster_results
            .drop(columns=["cluster"])
            .mean()
            .to_dict()
        )

        global_metrics["model"] = "Clustered_LightGBM"
        self.results.append(global_metrics)

    # ------------------------------------------------------------------
    # 3️⃣ LSTM models
    # ------------------------------------------------------------------
    def run_lstm(self, sequence_length=12, epochs=20):
        lstm = LSTMModels(sequence_length=sequence_length)

        X, y = lstm.build_sequences(
            self.df,
            feature_cols=self.feature_cols,
            target_col=self.target_col
        )

        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        X_train, X_val, X_test = (
            X[:train_end],
            X[train_end:val_end],
            X[val_end:]
        )

        y_train, y_val, y_test = (
            y[:train_end],
            y[train_end:val_end],
            y[val_end:]
        )

        model = lstm.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs
        )

        y_pred = model.predict(X_test).ravel()

        metrics = {
            "model": "LSTM",
            "MAE": ForecastMetrics.mae(y_test, y_pred),
            "RMSE": ForecastMetrics.rmse(y_test, y_pred),
            "SMAPE": ForecastMetrics.smape(y_test, y_pred),
            "WAPE": ForecastMetrics.wape(y_test, y_pred)
        }

        self.results.append(metrics)

    # ------------------------------------------------------------------
    # Final benchmark table
    # ------------------------------------------------------------------
    def get_results(self):
        return pd.DataFrame(self.results).sort_values("WAPE")

