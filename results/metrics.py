import pandas as pd


class ResultsManager:
    """
    Handle, aggregate and export forecasting results.

    This class is responsible for:
    - Collecting metrics from multiple models
    - Ranking models by performance
    - Exporting results for reporting
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Parameters
        ----------
        results_df : pd.DataFrame
            Output of ForecastingPipeline.get_results()
        """
        self.df = results_df.copy()

        required_cols = {"model", "MAE", "RMSE", "SMAPE", "WAPE"}
        missing = required_cols - set(self.df.columns)

        if missing:
            raise ValueError(f"Missing required metrics columns: {missing}")

    # ------------------------------------------------------------------
    # Ranking & comparison
    # ------------------------------------------------------------------
    def rank_models(self, metric="WAPE", ascending=True):
        """
        Rank models based on a selected metric.
        Default = WAPE (lower is better)
        """
        return (
            self.df
            .sort_values(metric, ascending=ascending)
            .reset_index(drop=True)
        )

    def best_model(self, metric="WAPE"):
        """
        Return the best performing model.
        """
        ranked = self.rank_models(metric=metric)
        return ranked.iloc[0]

    def summary_table(self, round_decimals=3):
        """
        Clean summary table for reporting.
        """
        return (
            self.df
            .set_index("model")
            .round(round_decimals)
            .sort_values("WAPE")
        )

    # ------------------------------------------------------------------
    # Aggregations & diagnostics
    # ------------------------------------------------------------------
    def compare_families(self):
        """
        Compare model families (ML, DL, Hybrid)
        based on naming convention.
        """
        def infer_family(model_name):
            if "LSTM" in model_name:
                return "Deep Learning"
            if "Cluster" in model_name:
                return "Hybrid (Clustering + ML)"
            return "Classical ML"

        df = self.df.copy()
        df["family"] = df["model"].apply(infer_family)

        return (
            df
            .groupby("family")[["MAE", "RMSE", "SMAPE", "WAPE"]]
            .mean()
            .round(3)
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_csv(self, path="results/model_metrics.csv"):
        """
        Save metrics to CSV.
        """
        self.df.to_csv(path, index=False)

    def save_summary(self, path="results/model_metrics_summary.csv"):
        """
        Save ranked summary table.
        """
        self.summary_table().to_csv(path)
