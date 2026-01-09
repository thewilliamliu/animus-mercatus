import pandas as pd
import numpy as np
from src.data_loader import load_data
from src.data_generator import generate_data
from src.features import DataFeatures
from src.model import CryptoPredictor

# Method that asks the user which data source to input for the model/results.
def get_data_choice():
    print("\n" + "="*50)
    print("Select your data source!")
    print("="*50)
    print("1. Generate synthetic data")
    print("2. Use sample dataset (used for README findings)")
    print("3. Use custom dataset")
    print("="*50)

    # Returns choice if user has responded correctly.
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice in ['1','2','3']:
            return choice;
        else:
            print("Invalid choice. Please enter 1, 2, or 3!")


choice = get_data_choice()
print("="*50)

# Generating data if choice == 1.
if choice == '1':
    df = generate_data()
    print(f"Generated {len(df)} days of data. The first 5 rows are displayed here.")

if choice == '2':
    df = load_data('sample_data.csv')
    print(f"Used sample data with {len(df)} days of data. The first 5 rows are displayed here.")

if choice == '3':
    filename = input("Enter your filename (with '.csv') in data/ folder: ").strip()
    df = load_data(filename)
    print(f"Used uploaded custom data with {len(df)} days of data. The first 5 rows are displayed here.")

# Displays the 5 rows.
print(df.head(5))

# Prepare features on the data.
data = DataFeatures()
df = data.prepare_features(df, n_lags=3)
print(f"Features prepared. Shape: {df.shape}")

# Splits data.
data.split_data(df, train_frac=0.7, val_frac=0.15)
print(f"Split data successfully. Train: {len(data.X_train)}, Validate: {len(data.X_val)}, Test: {len(data.X_test)}")

# Train the model using train data.
model = CryptoPredictor(n_estimators=100, random_state=42)
model.train(
    data.X_train, 
    data.y_train_direction, 
    data.y_train_volatility
)
print("Model trained!")

# Evaluate/test on the test data.
results = model.evaluate(
    data.X_test,
    data.y_test_direction,
    data.y_test_volatility
)

print("\n" + "="*50)
print("PERFORMANCE METRICS of ANIMUS-MERCATUS")
print("="*50)
print(f"Direction Accuracy: {results['direction_accuracy']:.3f}")
print(f"Direction F1 Score: {results['direction_f1']:.3f}")
print(f"Baseline Accuracy: {results['baseline_accuracy']:.3f}")
print(f"Beat Baseline: {results['beat_baseline']}")
print(f"Hit Rate (Up predictions): {results['hit_rate']:.3f}")
print(f"\nVolatility RMSE: {results['volatility_rmse']:.4f}")
print(f"Volatility MAE: {results['volatility_mae']:.4f}")