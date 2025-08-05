import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data/processed', 'models', 'reports/figures', 'eda']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Created necessary directories")

def clean_data(input_file='saudi_cars.csv'):
    """Clean and preprocess the data."""
    print("🔍 Loading and cleaning data...")
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # 1. Handle missing values
    print("   - Handling missing values...")
    df.dropna(inplace=True)
    
    # 2. Convert data types
    print("   - Converting data types...")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['kilometers'] = df['kilometers'].astype(str).str.replace(',', '').str.replace(' ', '').astype(float)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # 3. Remove unrealistic values
    print("   - Removing unrealistic values...")
    df = df[df['price'].between(1000, 2000000)]
    df = df[df['kilometers'].between(0, 500000)]
    df = df[df['year'].between(1990, datetime.now().year + 1)]
    
    # 4. Feature engineering
    print("   - Engineering features...")
    df['age'] = datetime.now().year - df['year']
    df['km_per_year'] = df['kilometers'] / (df['age'] + 1)
    
    # 5. Save processed data
    processed_path = 'data/processed/cleaned_saudi_cars.csv'
    df.to_csv(processed_path, index=False)
    print(f"💾 Saved processed data to {processed_path}")
    
    return df

def train_model(df):
    """Train and save the prediction model."""
    print("🤖 Training model...")
    
    # Encode categorical variables
    print("   - Encoding categorical variables...")
    categorical_cols = ['brand', 'model', 'fuel_type', 'gear_type', 'car_condition']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
        # Save the encoders
        encoder_path = f'models/le_{col}.pkl'
        joblib.dump(le, encoder_path)
        print(f"   - Saved {encoder_path}")
    
    # Prepare features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    print("   - Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   - MAE: {mae:.2f}")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - R² Score: {r2:.4f}")
    
    # Save the model
    model_path = 'models/car_price_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Saved model to {model_path}")
    
    # Save feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importances.to_csv('reports/feature_importances.csv', index=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importances.png')
    plt.close()
    
    return model, label_encoders

def main():
    # Create necessary directories
    create_directories()
    
    # Clean and preprocess the data
    df = clean_data()
    
    # Train and save the model
    model, label_encoders = train_model(df)
    
    print("\n🎉 Pipeline completed successfully!")
    print("You can now run the Streamlit app using: streamlit run app.py")

if __name__ == "__main__":
    main()
