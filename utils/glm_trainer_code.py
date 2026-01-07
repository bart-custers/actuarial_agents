import joblib
from sklearn.linear_model import PoissonRegressor

class GLMTrainer:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        self.model = PoissonRegressor(alpha=1e-6, max_iter=500)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        """Save the trained model."""
        if self.model:
            joblib.dump(self.model, path)
        else:
            raise ValueError("No trained model to save.")