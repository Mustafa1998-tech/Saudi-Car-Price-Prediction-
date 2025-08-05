import os
import sys
import traceback

print("=== Starting Debug Script ===")

# Check Python version
print(f"Python version: {sys.version}")

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if required files exist
required_files = [
    'models/car_price_model.pkl',
    'models/le_brand.pkl',
    'models/le_car_condition.pkl',
    'models/le_fuel_type.pkl',
    'models/le_gear_type.pkl',
    'models/le_model.pkl',
    'data/processed/cleaned_saudi_cars.csv'
]

print("\n=== Checking Required Files ===")
for file in required_files:
    exists = os.path.exists(file)
    print(f"{file}: {'✅ Found' if exists else '❌ Missing'}")

# Try to load the data and model
print("\n=== Testing Data Loading ===")
try:
    import pandas as pd
    import joblib
    
    # Test loading the data
    print("\nLoading data...")
    df = pd.read_csv('data/processed/cleaned_saudi_cars.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Test loading the model
    print("\nLoading model...")
    model = joblib.load('models/car_price_model.pkl')
    print("Model loaded successfully.")
    
    # Test loading encoders
    print("\nLoading encoders...")
    encoders = {}
    for col in ['brand', 'model', 'fuel_type', 'gear_type', 'car_condition']:
        encoders[col] = joblib.load(f'models/le_{col}.pkl')
        print(f"Loaded encoder for {col}")
    
    print("\n✅ All tests passed successfully!")
    
except Exception as e:
    print("\n❌ Error during testing:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print("\nStack trace:")
    traceback.print_exc()

print("\n=== Debug Script Finished ===")
