import os
from datetime import datetime
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
    
    def calibration_plot(self, y_true, y_pred, exposure, model_type="model"):
        """
        Creates a calibration/lift plot comparing actual vs predicted claim rates
        across prediction deciles.

        Parameters
        ----------
        y_true : array-like
            Observed claim counts.
        y_pred : array-like
            Model-predicted claim counts.
        exposure : array-like
            Exposure values for each observation.
        model_label : str
            Used for plot title and filename.

        Returns
        -------
        decile_summary : pd.DataFrame
            Table containing decile-level aggregated metrics.
        """

        # --- Compute rates ---
        actual_rate = np.asarray(y_true) / np.asarray(exposure)
        predicted_rate = np.asarray(y_pred) / np.asarray(exposure)

        df = pd.DataFrame({
            "ActualRate": actual_rate,
            "PredictedRate": predicted_rate,
            "Exposure": exposure
        })

        # Assign deciles based on predicted rate
        df["Decile"] = pd.qcut(df["PredictedRate"], q=10, labels=False, duplicates="drop")

        # Aggregate metrics by decile
        decile_summary = (
            df.groupby("Decile")
            .agg({
                "ActualRate": "mean",
                "PredictedRate": "mean",
                "Exposure": "sum"
            })
            .reset_index()
        )

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(decile_summary["Decile"], decile_summary["ActualRate"],
                marker="o", label="Actual Rate")
        ax1.plot(decile_summary["Decile"], decile_summary["PredictedRate"],
                marker="x", label="Predicted Rate")
        ax1.set_xlabel("Decile (by predicted rate)")
        ax1.set_ylabel("Average Claim Rate")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.bar(decile_summary["Decile"], decile_summary["Exposure"],
                alpha=0.3, width=0.8, label="Total Exposure")
        ax2.set_ylabel("Total Exposure")
        ax2.legend(loc="upper right")

        plt.title(f"Calibration Plot ({model_type.upper()})")
        plt.tight_layout()

        # --- Save plot ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "data/evaluation"
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f"calibration_{model_type}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close(fig)

        return decile_summary                

    def prediction_comparison_features(
        self,
        X_matrix,
        feature_names,
        preds_current,
        preds_previous,
        set_name,
        model_type):

        X = pd.DataFrame(X_matrix, columns=feature_names)
        X["_pred_cur"] = preds_current
        X["_pred_prev"] = preds_previous

        rows = []

        for feature in feature_names:
            try:
                grouped = (
                    X.groupby(feature)[["_pred_cur", "_pred_prev"]]
                    .mean()
                    .reset_index()
                )
            except Exception:
                continue

            grouped["diff"] = grouped["_pred_cur"] - grouped["_pred_prev"]
            grouped["abs_diff"] = grouped["diff"].abs()

            top_rows = grouped.nlargest(5, "abs_diff")

            for _, r in top_rows.iterrows():
                rows.append({
                    "Feature": feature,
                    "Value": r[feature],
                    "Prev_Pred": r["_pred_prev"],
                    "Cur_Pred": r["_pred_cur"],
                    "Diff": r["diff"],
                    "AbsDiff": r["abs_diff"],
                })

        deviation_table = pd.DataFrame(rows).sort_values("AbsDiff", ascending=False)

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
                color="salmon"
            )

            ax.set_title(f"{set_name.capitalize()} Predictions by {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Mean Prediction")
            ax.grid(True)
            ax.legend()

        # Remove any unused axes
        for j in range(i + 1, n_rows * n_cols):
            fig.delaxes(axes[j])

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "data/evaluation"
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(
            results_dir, f"prediction_comparison_{model_type}_{set_name}_{timestamp}.png"
        )

        plt.savefig(plot_path)
        plt.close(fig)

        return deviation_table
    
    def actual_vs_predicted_features(
        self,
        X_matrix,
        feature_names,
        preds_current,
        y_true,
        set_name,
        model_type):

        X = pd.DataFrame(X_matrix, columns=feature_names)
        X["_pred_cur"] = preds_current
        X["_actual"] = y_true

        rows = []

        for feature in feature_names:
            try:
                grouped = (
                    X.groupby(feature)[["_pred_cur", "_actual"]]
                    .mean()
                    .reset_index()
                )
            except Exception:
                continue

            grouped["diff"] = grouped["_pred_cur"] - grouped["_actual"]
            grouped["abs_diff"] = grouped["diff"].abs()

            top_rows = grouped.nlargest(5, "abs_diff")

            for _, r in top_rows.iterrows():
                rows.append({
                    "Feature": feature,
                    "Value": r[feature],
                    "Cur_Pred": r["_pred_cur"],
                    "Actual": r["_actual"],
                    "Diff": r["diff"],
                    "AbsDiff": r["abs_diff"],
                })

        deviation_table = pd.DataFrame(rows).sort_values("AbsDiff", ascending=False)

        n_features = len(feature_names)
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(feature_names):
            ax = axes[i]
            try:
                grouped = (
                    X.groupby(feature)[["_pred_cur", "_actual"]]
                    .mean()
                    .reset_index()
                    .sort_values(feature)
                )
            except Exception:
                continue

            ax.plot(grouped[feature], grouped["_pred_cur"], marker="o", label="Current Model")
            ax.plot(grouped[feature], grouped["_actual"], marker="x", label="Actual")

            ax.fill_between(
                grouped[feature],
                grouped["_pred_cur"],
                grouped["_actual"],
                alpha=0.3,
                color="green",
                label="Curr - Actual"
            )

            ax.set_title(f"{set_name}: {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Mean Prediction / Actual")
            ax.grid(True)
            ax.legend()

        # Remove unused axes
        for j in range(i+1, n_rows*n_cols):
            fig.delaxes(axes[j])

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "data/evaluation"
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f"Act_vs_Exp_{model_type}_{set_name}_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close(fig)

        return deviation_table

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
        metrics["Calibration_table"] = self.calibration_plot(y_true, y_pred, exposure, self.model_type)

        return metrics

    def evaluate_predicted(self,
        X_train, X_test,
        preds_train_current, preds_train_previous,
        preds_test_current, preds_test_previous,
        feature_names):

        # Checks
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)

        # Create tables with comparisons of predictions per feature
        train_preds_comparison = self.prediction_comparison_features(
            X_train, feature_names,
            preds_train_current, preds_train_previous,
            set_name="train",
            model_type=self.model_type)

        test_preds_comparison = self.prediction_comparison_features(
            X_test, feature_names,
            preds_test_current, preds_test_previous,
            set_name="test",
            model_type=self.model_type)

        return {
            "train_preds_comparison": train_preds_comparison,
            "test_preds_comparison": test_preds_comparison}
    
    def evaluate_act_vs_exp(self,
        X_train, X_test,
        preds_train_current, y_train,
        preds_test_current, y_test,
        feature_names):

        # Checks
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)

        # Create tables with comparisons of predictions per feature
        train_act_vs_exp = self.actual_vs_predicted_features(
            X_train, feature_names,
            preds_train_current, y_train,
            set_name="train",
            model_type=self.model_type)

        test_act_vs_exp = self.actual_vs_predicted_features(
            X_test, feature_names,
            preds_test_current, y_test,
            set_name="test",
            model_type=self.model_type)

        return {
            "train_act_vs_exp": train_act_vs_exp,
            "test_act_vs_exp": test_act_vs_exp}