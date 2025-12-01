import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataPipeline:
    """
    Executes a predefined insurance data cleaning and preprocessing routine.
    """

    def __init__(self):
        self.actions_log = []
        self.feature_names = None
        self.preprocessor = None

    def process(self, data: pd.DataFrame):
        """
        Perform deterministic data preprocessing.
        Returns:
            dict with cleaned DataFrames and metadata
        """
        # --- Step 1: Define features ---
        exposure = data['Exposure']
        y = data['ClaimNb']/data["Exposure"]
        y = y.clip(upper=10)
        X = data.drop(columns=['ClaimNb', 'Exposure', 'IDpol'], errors='ignore')

        numerical_features = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Area', 'Density']
        categorical_features = ['VehBrand', 'Region', 'VehGas']
        self.actions_log.append(f"Defined {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")

        # --- Step 2: Split dataset ---
        X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
            X, y, exposure, test_size=0.1, random_state=42
        )
        self.actions_log.append(f"Split data: {len(X_train)} train / {len(X_test)} test observations.")

        # --- Step 3: Preprocessing pipeline ---
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        X_train_prep = self.preprocessor.fit_transform(X_train)
        X_test_prep = self.preprocessor.transform(X_test)

        # --- Step 4: Extract transformed feature names ---
        cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
        self.feature_names = numerical_features + cat_feature_names
        self.actions_log.append(f"Generated {len(self.feature_names)} feature columns after encoding.")

        X_train_prep = np.asarray(X_train_prep)
        X_test_prep = np.asarray(X_test_prep)

        # Return all relevant components
        return {
            "X_train": X_train_prep,
            "X_test": X_test_prep,
            "y_train": y_train,
            "y_test": y_test,
            "exposure_train": exposure_train,
            "exposure_test": exposure_test,
            "feature_names": self.feature_names,
            "actions_log": self.actions_log,
        }

    def summary(self):
        """Human-readable summary of actions taken."""
        return "\n".join(self.actions_log)
