import os
import joblib
import pandas as pd

print("🔍 Verifying model and encoders...")

# Check if model file exists
model_path = 'models/car_price_model.pkl'
if not os.path.exists(model_path):
    print(f"❌ Model file not found at: {os.path.abspath(model_path)}")
else:
    try:
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded model from: {os.path.abspath(model_path)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Check encoder files
encoders = {
    'brand': 'models/le_brand.pkl',
    'model': 'models/le_model.pkl',
    'fuel_type': 'models/le_fuel_type.pkl',
    'gear_type': 'models/le_gear_type.pkl',
    'car_condition': 'models/le_car_condition.pkl'
}

for name, path in encoders.items():
    if not os.path.exists(path):
        print(f"❌ Encoder '{name}' not found at: {os.path.abspath(path)}")
    else:
        try:
            encoder = joblib.load(path)
            print(f"✅ Successfully loaded {name} encoder from: {os.path.abspath(path)}")
        except Exception as e:
            print(f"❌ Error loading {name} encoder: {e}")

# Check data file
data_path = 'data/processed/cleaned_saudi_cars.csv'
if not os.path.exists(data_path):
    print(f"❌ Processed data not found at: {os.path.abspath(data_path)}")
else:
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded data from: {os.path.abspath(data_path)}")
        print("\n📊 Data sample:")
        print(df.head())
    except Exception as e:
        print(f"❌ Error loading data: {e}")

print("\n🔍 Verification complete!")
