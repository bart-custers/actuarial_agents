import joblib
from sklearn.linear_model import PoissonRegressor

class GLMTrainer:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train,  exposure_train=None):
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

trainer = GLMTrainer()
trainer.train(X_train, y_train, exposure_train)
    
model = trainer.model
preds_train = trainer.predict(X_train)
preds_test = trainer.predict(X_test)

result = {"preds_train": preds_train, "preds_test": preds_test, "model": model}