import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

def load_data(filepath='data/processed/cleaned_saudi_cars.csv'):
    """Load and return the cleaned dataset."""
    if not os.path.exists(filepath):
        print("⚠️  Cleaned data not found. Running data cleaning first...")
        from data_cleaning import clean_data
        return clean_data()
    return pd.read_csv(filepath)

def plot_price_distribution(df):
    """Plot distribution of car prices."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['price'], bins=30, kde=True)
    plt.title('Distribution of Car Prices (SAR)', fontsize=15)
    plt.xlabel('Price (SAR)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda/price_distribution.png')
    plt.close()

def plot_price_by_brand(df):
    """Plot average price by car brand."""
    plt.figure(figsize=(14, 6))
    brand_avg = df.groupby('brand')['price'].mean().sort_values(ascending=False)
    sns.barplot(x=brand_avg.index, y=brand_avg.values)
    plt.title('Average Price by Brand', fontsize=15)
    plt.xlabel('Brand')
    plt.ylabel('Average Price (SAR)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('eda/price_by_brand.png')
    plt.close()

def plot_price_vs_year(df):
    """Plot price vs. year of manufacture."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='year', y='price', hue='brand', alpha=0.7)
    plt.title('Price vs. Year of Manufacture', fontsize=15)
    plt.xlabel('Year')
    plt.ylabel('Price (SAR)')
    plt.tight_layout()
    plt.savefig('eda/price_vs_year.png')
    plt.close()

def plot_price_vs_kilometers(df):
    """Plot price vs. kilometers driven."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='kilometers', y='price', hue='brand', alpha=0.6)
    plt.title('Price vs. Kilometers Driven', fontsize=15)
    plt.xlabel('Kilometers')
    plt.ylabel('Price (SAR)')
    plt.tight_layout()
    plt.savefig('eda/price_vs_kilometers.png')
    plt.close()

def plot_fuel_type_impact(df):
    """Plot impact of fuel type on price."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='fuel_type', y='price')
    plt.title('Price Distribution by Fuel Type', fontsize=15)
    plt.xlabel('Fuel Type')
    plt.ylabel('Price (SAR)')
    plt.tight_layout()
    plt.savefig('eda/price_by_fuel_type.png')
    plt.close()

def generate_correlation_heatmap(df):
    """Generate correlation heatmap for numerical features."""
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap', fontsize=15)
    plt.tight_layout()
    plt.savefig('eda/correlation_heatmap.png')
    plt.close()

def generate_eda_report(df):
    """Generate a comprehensive EDA report."""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'total_samples': len(df),
            'numerical_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'basic_stats': df.describe().to_dict()
        },
        'price_analysis': {
            'average_price': int(df['price'].mean()),
            'median_price': int(df['price'].median()),
            'min_price': int(df['price'].min()),
            'max_price': int(df['price'].max()),
            'price_std': int(df['price'].std())
        },
        'popular_brands': df['brand'].value_counts().to_dict(),
        'fuel_type_distribution': df['fuel_type'].value_counts().to_dict(),
        'gear_type_distribution': df['gear_type'].value_counts().to_dict()
    }
    
    with open('eda/eda_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report

def main():
    # Create output directory
    os.makedirs('eda', exist_ok=True)
    
    # Load data
    print("🔍 Loading data...")
    df = load_data()
    
    # Generate plots
    print("📊 Generating visualizations...")
    plot_price_distribution(df)
    plot_price_by_brand(df)
    plot_price_vs_year(df)
    plot_price_vs_kilometers(df)
    plot_fuel_type_impact(df)
    generate_correlation_heatmap(df)
    
    # Generate report
    print("📝 Generating EDA report...")
    report = generate_eda_report(df)
    
    # Print summary
    print("\n📊 EDA Summary:")
    print(f"   - Total samples: {report['dataset_info']['total_samples']}")
    print(f"   - Average price: {report['price_analysis']['average_price']:,} SAR")
    print(f"   - Most popular brand: {max(report['popular_brands'].items(), key=lambda x: x[1])[0]}")
    print(f"   - Most common fuel type: {max(report['fuel_type_distribution'].items(), key=lambda x: x[1])[0]}")
    print("\n✨ EDA completed successfully! Check the 'eda' folder for visualizations and report.")

if __name__ == "__main__":
    main()
