import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib

# Create necessary directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

def clean_data(filepath='saudi_cars.csv'):
    """
    Clean and preprocess the Saudi car market data.
    
    Args:
        filepath (str): Path to the raw data CSV file
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    print("🔍 Loading and cleaning data...")
    
    # Load the data
    df = pd.read_csv(filepath)
    
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
    df = df[df['price'].between(1000, 2000000)]  # Reasonable price range
    df = df[df['kilometers'].between(0, 500000)]  # Reasonable mileage range
    df = df[df['year'].between(1990, datetime.now().year + 1)]  # Reasonable year range
    
    # 4. Feature engineering
    print("   - Engineering features...")
    df['age'] = datetime.now().year - df['year']
    df['price_per_km'] = df['price'] / (df['kilometers'] + 1)  # Add 1 to avoid division by zero
    
    # 5. Save processed data
    print("💾 Saving processed data...")
    df.to_csv('data/processed/cleaned_saudi_cars.csv', index=False)
    
    # 6. Save metadata
    metadata = {
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': len(df),
        'columns': list(df.columns),
        'brands': df['brand'].unique().tolist(),
        'fuel_types': df['fuel_type'].unique().tolist(),
        'gear_types': df['gear_type'].unique().tolist(),
        'min_year': int(df['year'].min()),
        'max_year': int(df['year'].max()),
        'min_price': int(df['price'].min()),
        'max_price': int(df['price'].max()),
    }
    
    with open('data/processed/metadata.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print("✅ Data cleaning completed successfully!")
    return df

if __name__ == "__main__":
    # Run data cleaning
    df = clean_data()
    
    # Print summary
    print("\n📊 Data Summary:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Brands: {', '.join(df['brand'].unique())}")
    print(f"   - Price range: {df['price'].min():,} - {df['price'].max():,} SAR")
    print(f"   - Year range: {int(df['year'].min())} - {int(df['year'].max())}")
    print("\n✨ Data is ready for analysis and modeling! ✨")
