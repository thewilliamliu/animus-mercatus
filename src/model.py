import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error    

import data_generator

class CryptoPredictor:

    # For each object, assigns a trained classifier with 100 trees in the forest.
    def __init__(self, n_estimators=100, random_state=42):
        self.direction_model = RandomForestClassifier(
            n_estimators = n_estimators,
            random_state=random_state
        )
        self.volatility_model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )

    # Here, X represents the inputs (from Reddit + YFinance). On the output end, y represents the labels.
    def train(self, X, y_direction, y_volatility):
        self.feature_names = X.columns # Saves column names for later.
        self.direction_model.fit(X, y_direction)
        self.volatility_model.fit(X, y_volatility)

    # Predicts either 0 or 1 depending on if the market will likely move down or up.
    def predict_direction(self, X):
        return self.direction_model.predict(X)

    # Predicts a percentage for the next market day's volatility.
    def predict_volatility(self, X):
        return self.volatility_model.predict(X)

    # Method to return each feature's importance.
    def feature_importance(self, X):
        return {
            'direction': dict(zip(self.feature_names, self.direction_model.feature_importances_)),
            'volatility': dict(zip(self.feature_names, self.volatility_model.feature_importances_))
        }

    # Evaluates results/testing data. Returns a variety of metrics that are all explained in the README file.
    def evaluate(self, X, y_direction, y_volatility):

        # Get predictions using model we programmed above.
        direction_pred = self.predict_direction(X)
        volatility_pred = self.predict_volatility(X)
         
        # Accuracy: computes how well the model actually did.
        accuracy = accuracy_score(y_direction, direction_pred)

        # F1: Harmonic mean of precision and recall. Catches the "missed" cases. Explained more in README.
        f1 = f1_score(y_direction, direction_pred)
        
        # Baseline: comparing our model to the dumbest possible model. Does it perform better/worse?
        # The baseline model in this case will always predict the most frequent value in past data. 
        baseline_pred = [y_direction.mode()[0]] * len(y_direction)
        baseline_accuracy = accuracy_score(y_direction, baseline_pred)
        beat_baseline = accuracy > baseline_accuracy

        # Hit Rate: % of up predictions were actually up. Only isolates accuracy of "buy" signals.
        correct_ups = sum((direction_pred == 1) & (y_direction == 1))
        total_up_predictions = sum(direction_pred == 1)
        hit_rate = correct_ups / total_up_predictions if total_up_predictions > 0 else 0
    
        # Volatility Metrics: simply returns RSME and MAE error types.
        volatility_rmse = np.sqrt(mean_squared_error(y_volatility, volatility_pred))
        volatility_mae = mean_absolute_error(y_volatility, volatility_pred)
        
        return {
            'direction_accuracy': accuracy,
            'direction_f1': f1,
            'baseline_accuracy': baseline_accuracy,
            'beat_baseline': beat_baseline,
            'hit_rate': hit_rate,
            'volatility_rmse': volatility_rmse,
            'volatility_mae': volatility_mae
        }