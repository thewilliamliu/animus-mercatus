import pandas as pd
import numpy as np

class DataFeatures:
    # Prepares the data for actual modelling. n-lags examines how many past days the model can examine.
    def prepare_features(self, df, n_lags=3):
        df = df.copy() # Defensive copy.

        # Binary classifier to say if the price went up (1) or down (0) tomorrow.
        df['target_direction'] = (df['return'].shift(-1) > 0).astype(int) 
        # Regressive classifier to s
        df['target_volatility'] = df['return'].shift(-1).abs()

        # Shifting back stuff.
        for lag in range(1, n_lags + 1):
            df[f'return_lag_{lag}'] = df['return'].shift(lag)
            df[f'sentiment_lag_{lag}'] = df['avg_sentiment'].shift(lag)

        df.dropna(inplace=True) # Gets rid of any NaN rows that are useless.

        return df

    # Splits the data into training data, validation data, and testing data.
    def split_data(self, df, train_frac = 0.7, val_frac=0.15):
        n = len(df)

        # Defining cutoff points in time.
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        # Primes the X and y we'll later distinguish in ML. 
        X = df.drop(columns=['target_direction', 'target_volatility', 'date', 'price']) 
        y_direction = df['target_direction']
        y_volatility = df['target_volatility']
        
        # Training data.
        self.X_train = X.iloc[:train_end]
        self.y_train_direction = y_direction.iloc[:train_end]
        self.y_train_volatility = y_volatility.iloc[:train_end]

        # Validation data.
        self.X_val = X.iloc[train_end:val_end]
        self.y_val_direction = y_direction.iloc[train_end:val_end]
        self.y_val_volatility = y_volatility.iloc[train_end:val_end]

        # Testing data.
        self.X_test = X.iloc[val_end:]
        self.y_test_direction = y_direction.iloc[val_end:]
        self.y_test_volatility = y_volatility.iloc[val_end:]

        