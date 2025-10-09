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

    def clean(self, data: pd.DataFrame):
        """
        Perform deterministic data cleaning and preprocessing.
        Returns:
            dict with cleaned DataFrames and metadata
        """

        # --- Step 1: Clip outliers / right-censoring ---
        self.actions_log.append("Clipping VehAge (<=20), DrivAge (<=90), BonusMalus (<=150), ClaimNb (<=5).")
        data['VehAge'] = data['VehAge'].clip(upper=20)
        data['DrivAge'] = data['DrivAge'].clip(upper=90)
        data['BonusMalus'] = data['BonusMalus'].clip(upper=150)
        data['ClaimNb'] = data['ClaimNb'].clip(upper=5)

        # --- Step 2: Encode Area as ordinal numeric ---
        area_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
        data['Area'] = data['Area'].map(area_mapping)
        self.actions_log.append("Mapped Area categories to numeric scale (A→1,...,F→6).")

        # --- Step 3: Drop missing values ---
        before = len(data)
        data = data.dropna()
        after = len(data)
        self.actions_log.append(f"Dropped {before - after} rows with missing values.")

        # --- Step 4: Define features ---
        exposure = data['Exposure']
        y = data['ClaimNb']
        X = data.drop(columns=['ClaimNb', 'Exposure', 'IDpol'], errors='ignore')

        numerical_features = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Area']
        categorical_features = ['VehBrand', 'Region', 'VehGas']
        self.actions_log.append(f"Defined {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")

        # --- Step 5: Split dataset ---
        X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
            X, y, exposure, test_size=0.1, random_state=42
        )
        self.actions_log.append(f"Split data: {len(X_train)} train / {len(X_test)} test observations.")

        # --- Step 6: Preprocessing pipeline ---
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        X_train_prep = self.preprocessor.fit_transform(X_train)
        X_test_prep = self.preprocessor.transform(X_test)

        # --- Step 7: Extract transformed feature names ---
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
