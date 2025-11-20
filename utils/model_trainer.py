import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, model_type="glm", offset=None):
        """
        model_type : str
            "glm" (Poisson GLM) or "gbm"
        offset : array-like or None
            Optional offset (e.g., log(Exposure)) for GLM
        """
        self.model_type = model_type
        self.offset = offset
        self.model = None

    def train(self, X_train, y_train, exposure_train):
        """Fit model according to selected type."""
        if self.model_type == "glm":
            self.model = PoissonRegressor(alpha=1e-6, max_iter=500)
            self.model.fit(X_train, y_train)

        elif self.model_type == "gbm":

            gbm = HistGradientBoostingRegressor(random_state=42, loss="poisson", verbose=1)

            # Small grid just for testing
            param_grid = {
                "max_iter": [200, 300],
                "learning_rate": [0.01, 0.05],
                "max_depth": [4, 5]
            }

            search = GridSearchCV(
                estimator=gbm,
                param_grid=param_grid,
                scoring="neg_mean_poisson_deviance",
                cv=3,
                n_jobs=-1,
                verbose=2
            )

            search.fit(X_train, y_train, sample_weight=exposure_train)

            self.model = search.best_estimator_

        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' not supported yet."
            )

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
