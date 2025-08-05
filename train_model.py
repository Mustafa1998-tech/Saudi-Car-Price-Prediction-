import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

def load_data():
    """Load and preprocess the car price data."""
    # Define the path to your dataset
    data_paths = [
        'data/processed/cleaned_saudi_cars.csv',
        'cleaned_saudi_cars.csv',
        os.path.join(os.path.dirname(__file__), 'data', 'processed', 'cleaned_saudi_cars.csv'),
        os.path.join(os.path.dirname(__file__), 'cleaned_saudi_cars.csv')
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"Data loaded successfully from: {os.path.abspath(path)}")
            break
    
    if df is None:
        print("Error: Could not find the dataset file. Please ensure you have the 'cleaned_saudi_cars.csv' file.")
        return None
    
    return df

def preprocess_data(df):
    """Preprocess the data for training."""
    # Make a copy of the dataframe
    df = df.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Add age feature
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    
    # Add km_per_year feature
    df['km_per_year'] = df['kilometers'] / (df['age'] + 1)  # +1 to avoid division by zero
    
    # Select features and target
    features = ['brand', 'model', 'year', 'fuel_type', 'kilometers', 'gear_type', 'car_condition', 'age', 'km_per_year']
    target = 'price'
    
    # Ensure all required columns exist
    for col in features + [target]:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in the dataset.")
            return None, None, None
    
    X = df[features]
    y = df[target]
    
    return X, y, features

def train_model(X, y, features):
    """Train the model and save it along with the encoders."""
    # Encode categorical variables
    encoders = {}
    X_encoded = X.copy()
    
    for col in ['brand', 'model', 'fuel_type', 'gear_type', 'car_condition']:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    return model, encoders

def save_model_and_encoders(model, encoders):
    """Save the trained model and encoders to disk."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = os.path.join('models', 'car_price_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to: {os.path.abspath(model_path)}")
    
    # Save the encoders
    encoders_path = os.path.join('models', 'encoders.pkl')
    joblib.dump(encoders, encoders_path)
    print(f"Encoders saved to: {os.path.abspath(encoders_path)}")

def main():
    print("🚗 Saudi Car Price Prediction Model Training")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y, features = preprocess_data(df)
    if X is None:
        return
    
    # Train model
    print("\nTraining model...")
    model, encoders = train_model(X, y, features)
    
    # Save model and encoders
    if model is not None and encoders is not None:
        print("\nSaving model and encoders...")
        save_model_and_encoders(model, encoders)
        print("\n✅ Model training and saving completed successfully!")
    else:
        print("\n❌ Model training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
