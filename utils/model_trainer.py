import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    """
    Train and manage actuarial frequency models using a unified interface.

    Currently supports:
    - Poisson GLM (for transparent, baseline models)
    - Gradient Boosting with Poisson loss (for non-linear extensions)
    """
    def __init__(self, model_type="glm", offset=None):
        """
        Args:
            model_type (str): Model family to train: "glm" or "gbm".
            offset (array-like or None): Optional exposure-based offset (e.g., log(Exposure)) for GLMs.
                Included for extensibility and consistency, even if not always used.
        """
        self.model_type = model_type
        self.offset = offset
        self.model = None

    def train(self, X_train, y_train, exposure_train):
        """
        Fit the selected model type on training data.

        Args:
            X_train (array-like): Design matrix after preprocessing.
            y_train (array-like): Target variable (e.g., claim frequency).
            exposure_train (array-like): Exposure values, used as sample weights for GBM.
        """
        if self.model_type == "glm":
            self.model = PoissonRegressor(alpha=1e-6, max_iter=500)
            self.model.fit(X_train, y_train)

        elif self.model_type == "gbm":

            gbm = HistGradientBoostingRegressor(random_state=42, loss="poisson", verbose=1)

            # Small grid just for testing
            param_grid = {
                "max_iter": [200],
                "learning_rate": [0.01, 0.05],
                "max_depth": [4]
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
        """
        Generate predictions using the trained model.

        Args:
            X (array-like): Feature matrix.

        Returns:
            np.ndarray: Model predictions.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)

    def save(self, path):
        """
        Persist the trained model to disk.

        Args:
            path (str): File path where the model will be saved.
        """
        if self.model:
            joblib.dump(self.model, path)
        else:
            raise ValueError("No trained model to save.")
