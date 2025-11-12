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

    def evaluate(self, y_true, y_pred, feature_names=None, exposure=None):
        """Compute evaluation metrics and plots."""
        metrics = {}

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

        # Core error metrics
        rmse = float(root_mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        metrics["RMSE"] = rmse
        metrics["MAE"] = mae
        metrics["R2"] = r2

        # Poisson deviance (only valid for count data)
        try:
            metrics["Poisson_Deviance"] = float(mean_poisson_deviance(y_true, y_pred))
        except Exception:
            metrics["Poisson_Deviance"] = None

        # Calibration / Lift plot (exposure optional)
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

            results_dir = "data/results"
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f"calibration_{self.model_type}.png")
            plt.savefig(plot_path)
            plt.close(fig)

            metrics["Calibration Plot"] = plot_path

        return metrics


# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_poisson_deviance

# class ModelEvaluation:
#     def __init__(self, model_type="glm", offset=None):
#         """
#         model_type : str
#             "glm" (Poisson GLM) or "gbm" or other custom models later
#         offset : array-like or None
#             Optional offset (e.g., log(Exposure)) for GLM
#         """
#         self.model_type = model_type
#         self.offset = offset
#         self.model = None

    
#     def evaluate(self, y_true, y_pred, feature_names,exposure=None):
#         """Compute evaluation metrics."""
#         # Core metrics and info
#         coefficients = self.model.coef_
#         coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
#         rmse = float(root_mean_squared_error(y_true, y_pred))
#         mae = float(mean_absolute_error(y_true, y_pred))  # Convert to float
#         poisson_dev = float(mean_poisson_deviance(y_true, y_pred))

#         metrics = {
#             "Coef": coef_df,
#             "RMSE": rmse,
#             "MAE": mae,
#             "Poisson deviance": poisson_dev,
#         }

#         # Plots
#         if exposure is not None:
#             actual_rate = y_true / exposure
#             predicted_rate = y_pred / exposure
#             df = pd.DataFrame({
#                 "ActualRate": actual_rate,
#                 "PredictedRate": predicted_rate,
#                 "Exposure": exposure
#             })
#             # Create deciles by predicted rate
#             df["Decile"] = pd.qcut(df["PredictedRate"], q=10, labels=False, duplicates="drop")

#             # Summarize per decile
#             decile_summary = (
#                 df.groupby("Decile")
#                 .agg({
#                     "ActualRate": "mean",
#                     "PredictedRate": "mean",
#                     "Exposure": "sum"
#                 })
#                 .reset_index()
#             )

#             # Plot
#             fig, ax1 = plt.subplots(figsize=(8, 5))
#             ax1.plot(decile_summary["Decile"], decile_summary["ActualRate"], marker="o", label="Actual Rate", color="blue")
#             ax1.plot(decile_summary["Decile"], decile_summary["PredictedRate"], marker="x", label="Predicted Rate", color="orange")
#             ax1.set_xlabel("Decile (by predicted rate)")
#             ax1.set_ylabel("Average Claim Rate")
#             ax1.legend(loc="upper left")
#             ax1.grid(True)

#             ax2 = ax1.twinx()
#             ax2.bar(decile_summary["Decile"], decile_summary["Exposure"], alpha=0.3, color="cornflowerblue", width=0.8, label="Total Exposure")
#             ax2.set_ylabel("Total Exposure")
#             ax2.legend(loc="upper right")

#             plt.title(f"Calibration Plot")
#             plt.tight_layout()

#             # Save figure
#             results_dir = "data/results"
#             os.makedirs(results_dir, exist_ok=True)
#             plot_path = os.path.join(results_dir, f"calibration.png")
#             plt.savefig(plot_path)
#             # Do not call plt.show() here; leave display to the caller to avoid
#             # duplicate rendering (the ModellingAgent displays the saved image).
#             plt.close(fig)

#             metrics["Calibration Plot"] = plot_path

#         return metrics
