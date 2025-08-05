import pandas as pd
from datetime import datetime
import os

def clean_and_prepare_data(df):
    """Clean and prepare the car data with consistent formatting."""
    # Make column names lowercase for consistency
    df.columns = df.columns.str.lower()
    
    # Ensure required columns exist
    required_columns = ['brand', 'model', 'year', 'kilometers', 'fuel_type', 'gear_type', 'car_condition', 'price']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean and standardize text columns
    text_columns = ['brand', 'model', 'fuel_type', 'gear_type', 'car_condition']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
    
    # Ensure price and kilometers are numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['kilometers'] = pd.to_numeric(df['kilometers'], errors='coerce')
    
    # Calculate age and km_per_year
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    df['km_per_year'] = df['kilometers'] / (df['age'] + 1)  # +1 to avoid division by zero
    
    # Remove any rows with missing critical data
    df = df.dropna(subset=['brand', 'model', 'year', 'price'])
    
    return df

def update_car_database(new_data_path, output_dir='data/processed'):
    """Update the car database with new data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cleaned_saudi_cars.csv')
    
    # Load existing data if it exists
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        print(f"Loaded existing data with {len(existing_df)} records")
    else:
        existing_df = pd.DataFrame()
        print("No existing data found, starting fresh")
    
    # Load new data
    new_df = pd.read_csv(new_data_path)
    print(f"Loaded new data with {len(new_df)} records")
    
    # Clean and prepare both datasets
    existing_cleaned = clean_and_prepare_data(existing_df) if not existing_df.empty else pd.DataFrame()
    new_cleaned = clean_and_prepare_data(new_df)
    
    # Combine datasets and remove duplicates
    combined_df = pd.concat([existing_cleaned, new_cleaned], ignore_index=True)
    
    # Remove duplicates based on key columns
    key_columns = ['brand', 'model', 'year', 'kilometers', 'fuel_type', 'gear_type', 'car_condition']
    combined_df = combined_df.drop_duplicates(subset=key_columns, keep='last')
    
    # Sort by brand, model, and year
    combined_df = combined_df.sort_values(by=['brand', 'model', 'year'], ascending=[True, True, False])
    
    # Save the updated dataset
    combined_df.to_csv(output_path, index=False)
    print(f"Saved updated dataset with {len(combined_df)} records to {output_path}")
    
    # Show some statistics
    print("\nUpdated dataset summary:")
    print(f"Total records: {len(combined_df)}")
    print(f"Brands: {combined_df['brand'].nunique()}")
    print(f"Models: {combined_df['model'].nunique()}")
    print(f"Years: {combined_df['year'].min()} - {combined_df['year'].max()}")
    print(f"Price range: {combined_df['price'].min():,.0f} - {combined_df['price'].max():,.0f} SAR")

if __name__ == "__main__":
    # Example usage:
    # 1. Save the new data to a file (e.g., new_cars.csv)
    # 2. Run: python update_car_data.py new_cars.csv
    
    import sys
    
    if len(sys.argv) > 1:
        new_data_file = sys.argv[1]
        if os.path.exists(new_data_file):
            update_car_database(new_data_file)
        else:
            print(f"Error: File not found: {new_data_file}")
    else:
        print("Usage: python update_car_data.py <path_to_new_data.csv>")
        print("\nPlease provide the path to the CSV file containing new car data.")
        print("The file should have the following columns:")
        print("brand, model, year, kilometers, fuel_type, gear_type, car_condition, price")
