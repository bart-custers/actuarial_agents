import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_poisson_deviance,
    r2_score,
)

class ModelEvaluation:
    def __init__(self, model=None, model_type="glm", offset=None):
        """
        model_type : str
            "glm" (Poisson GLM), "gbm", or other future models
        model : trained model object
        offset : array-like or None
            Optional offset (e.g., log(Exposure)) for GLM
        """
        self.model = model
        self.model_type = model_type.lower()
        self.offset = offset
    
    def gini_coefficient(y_true, y_pred):
        """Compute standard (non-normalized) Gini."""
        # Sort by predictions descending
        order = np.argsort(-y_pred)
        y_true_sorted = y_true[order]

        n = len(y_true)
        cum_y = np.cumsum(y_true_sorted)
        gini_sum = cum_y.sum() / cum_y[-1]  # normalize by total

        # Gini relative to uniform line
        return (2 * gini_sum / n) - (n + 1) / n

    def evaluate(self, y_true, y_pred, feature_names=None, exposure=None):
        """Compute evaluation metrics and plots."""
        metrics = {}

        # Core error metrics
        rmse = float(root_mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        metrics["RMSE"] = rmse
        metrics["MAE"] = mae
        metrics["R2"] = r2

        # Poisson deviance
        try:
            metrics["Poisson_Deviance"] = float(mean_poisson_deviance(y_true, y_pred))
        except Exception:
            metrics["Poisson_Deviance"] = None

        # Gini coefficient
        try:
            metrics["Gini_score"] = float(self.gini_coefficient(np.array(y_true), np.array(y_pred)))
        except Exception:
            metrics["Gini_score"] = None

        # Coefficients / Feature importance
        coef_df = None
        if hasattr(self.model, "coef_"):  # GLM-like models
            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": self.model.coef_.ravel()
            })
        elif hasattr(self.model, "feature_importances_"):  # GBM-like models
            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": self.model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
        else:
            coef_df = pd.DataFrame({"Feature": feature_names or [], "Weight": np.nan})
        
        metrics["Feature_Importance"] = coef_df

        # Calibration / Lift plot
        if exposure is not None:
            actual_rate = y_true / exposure
            predicted_rate = y_pred / exposure

            df = pd.DataFrame({
                "ActualRate": actual_rate,
                "PredictedRate": predicted_rate,
                "Exposure": exposure
            })
            df["Decile"] = pd.qcut(df["PredictedRate"], q=10, labels=False, duplicates="drop")

            decile_summary = (
                df.groupby("Decile")
                .agg({
                    "ActualRate": "mean",
                    "PredictedRate": "mean",
                    "Exposure": "sum"
                })
                .reset_index()
            )

            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(decile_summary["Decile"], decile_summary["ActualRate"],
                     marker="o", label="Actual Rate", color="blue")
            ax1.plot(decile_summary["Decile"], decile_summary["PredictedRate"],
                     marker="x", label="Predicted Rate", color="orange")
            ax1.set_xlabel("Decile (by predicted rate)")
            ax1.set_ylabel("Average Claim Rate")
            ax1.legend(loc="upper left")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.bar(decile_summary["Decile"], decile_summary["Exposure"],
                    alpha=0.3, color="cornflowerblue", width=0.8, label="Total Exposure")
            ax2.set_ylabel("Total Exposure")
            ax2.legend(loc="upper right")

            plt.title(f"Calibration Plot ({self.model_type.upper()})")
            plt.tight_layout()

            results_dir = "data/results/evaluation"
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f"calibration_{self.model_type}.png")
            plt.savefig(plot_path)
            plt.close(fig)

            metrics["Calibration Plot"] = plot_path

        return metrics

    def compare_prediction_deviations(X: pd.DataFrame,
                                    pred1: np.ndarray,
                                    pred2: np.ndarray,
                                    top_k: int = 3):
        """
        For each feature, computes average predictions by feature value
        and returns a table containing only the top absolute deviations.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset (train or test).
        pred1 : array-like
            Predictions from model 1.
        pred2 : array-like
            Predictions from model 2.
        top_k : int
            Number of largest absolute deviations to show per feature.

        Returns
        -------
        pd.DataFrame
            Table with largest absolute deviations.
        """

        df = X.copy()
        df["_pred_old"] = pred1
        df["_pred_new"] = pred2

        rows = []

        for feature in X.columns:
            try:
                grouped = (
                    df.groupby(feature)[["_pred_old", "_pred_new"]]
                    .mean()
                    .reset_index()
                )
            except Exception:
                continue

            grouped["diff"] = grouped["_pred_new"] - grouped["_pred_old"]
            grouped["abs_diff"] = grouped["diff"].abs()

            top_rows = grouped.nlargest(top_k, "abs_diff")

            for _, r in top_rows.iterrows():
                rows.append({
                    "Feature": feature,
                    "Value": r[feature],
                    "Old_Pred": r["_pred_old"],
                    "New_Pred": r["_pred_new"],
                    "Diff": r["diff"],
                    "AbsDiff": r["abs_diff"]
                })

        result = pd.DataFrame(rows)

        # Sort entire table by absolute deviation
        result = result.sort_values("AbsDiff", ascending=False).reset_index(drop=True)

        return result

    def plot_feature_comparisons(X_matrix, feature_names, preds_current, preds_previous, 
        set_name, model_type):
        """
        Compare current vs previous predictions by feature.
        Uses X_train / X_test.
        """

        X = pd.DataFrame(X_matrix, columns=feature_names)
        X["_pred_cur"] = preds_current
        X["_pred_prev"] = preds_previous

        n_features = len(feature_names)
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(feature_names):

            ax = axes[i]

            try:
                grouped = (
                    X.groupby(feature)[["_pred_cur", "_pred_prev"]]
                    .mean()
                    .reset_index()
                    .sort_values(feature)
                )
            except Exception:
                # Non-groupable features (e.g., high cardinality)
                continue

            ax.plot(
                grouped[feature], grouped["_pred_cur"],
                linestyle="--", marker="o", label="Current Model"
            )
            ax.plot(
                grouped[feature], grouped["_pred_prev"],
                linestyle="-", marker="s", label="Previous Model"
            )

            ax.fill_between(
                grouped[feature],
                grouped["_pred_cur"],
                grouped["_pred_prev"],
                alpha=0.3,
                color="salmon")

            ax.set_title(f"{set_name.capitalize()} Predictions by {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Mean Prediction")
            ax.grid(True)
            ax.legend()

        # Remove unused axes
        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axes[j])

        plt.tight_layout()

        results_dir = "data/results/evaluation"
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(
            results_dir, f"prediction_comparisons_{model_type}_{set_name}.png")        
        plt.savefig(out_path)
        plt.close(fig)
    
    