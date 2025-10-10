import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_poisson_deviance
from sklearn.linear_model import PoissonRegressor

class ModelTrainer:
    def __init__(self, model_type="glm", offset=None):
        """
        model_type : str
            "glm" (Poisson GLM) or "gbm" or other custom models later
        offset : array-like or None
            Optional offset (e.g., log(Exposure)) for GLM
        """
        self.model_type = model_type
        self.offset = offset
        self.model = None

    def train(self, X_train, y_train):
        """Fit model according to selected type."""
        if self.model_type == "glm":
            # Fit Poisson GLM (actuarial claim frequency model)
            self.model = PoissonRegressor(alpha=1e-6, max_iter=500)
            self.model.fit(X_train, y_train, sample_weight=None)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' not supported yet.")

    def predict(self, X):
        """Return predictions from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred, feature_names,exposure=None):
        """Compute evaluation metrics."""
        # Core metrics and info
        coefficients = self.model.coef_
        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
        rmse = float(root_mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))  # Convert to float
        poisson_dev = float(mean_poisson_deviance(y_true, y_pred))

        metrics = {
            "Coef": coef_df,
            "RMSE": rmse,
            "MAE": mae,
            "Poisson deviance": poisson_dev,
        }

        # Plots
        if exposure is not None:
            actual_rate = y_true / exposure
            predicted_rate = y_pred / exposure
            df = pd.DataFrame({
                "ActualRate": actual_rate,
                "PredictedRate": predicted_rate,
                "Exposure": exposure
            })
            # Create deciles by predicted rate
            df["Decile"] = pd.qcut(df["PredictedRate"], q=10, labels=False, duplicates="drop")

            # Summarize per decile
            decile_summary = (
                df.groupby("Decile")
                .agg({
                    "ActualRate": "mean",
                    "PredictedRate": "mean",
                    "Exposure": "sum"
                })
                .reset_index()
            )

            # Plot
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(decile_summary["Decile"], decile_summary["ActualRate"], marker="o", label="Actual Rate", color="blue")
            ax1.plot(decile_summary["Decile"], decile_summary["PredictedRate"], marker="x", label="Predicted Rate", color="orange")
            ax1.set_xlabel("Decile (by predicted rate)")
            ax1.set_ylabel("Average Claim Rate")
            ax1.legend(loc="upper left")
            ax1.grid(True)

            ax2 = ax1.twinx()
            ax2.bar(decile_summary["Decile"], decile_summary["Exposure"], alpha=0.3, color="cornflowerblue", width=0.8, label="Total Exposure")
            ax2.set_ylabel("Total Exposure")
            ax2.legend(loc="upper right")

            plt.title(f"Calibration Plot")
            plt.tight_layout()

            # Save figure
            plot_path = os.path.join(self.artifacts_dir, f"calibration.png")
            plt.savefig(plot_path)
            plt.show()
            plt.close(fig)

            metrics["Calibration Plot"] = plot_path

        return metrics

    def save(self, path):
        """Save the trained model."""
        if self.model:
            joblib.dump(self.model, path)
        else:
            raise ValueError("No trained model to save.")
