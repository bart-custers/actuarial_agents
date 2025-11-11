import os
import numpy as np
import pandas as pd
import joblib
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

    def save(self, path):
        """Save the trained model."""
        if self.model:
            joblib.dump(self.model, path)
        else:
            raise ValueError("No trained model to save.")
