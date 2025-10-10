import numpy as np
import pandas as pd
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

    def evaluate(self, y_true, y_pred, feature_names):
        """Compute evaluation metrics."""
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
        return metrics

    def save(self, path):
        """Save the trained model."""
        if self.model:
            joblib.dump(self.model, path)
        else:
            raise ValueError("No trained model to save.")
