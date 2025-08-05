import os
import sys
import time
import json
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error
)
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = 'models'
DATA_DIR = 'data/processed'
REPORTS_DIR = 'reports'
RANDOM_STATE = 42

# Create necessary directories
for directory in [MODEL_DIR, DATA_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class ModelTrainer:
    """A class to handle model training and evaluation."""
    
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'LightGBM': LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            'LinearRegression': LinearRegression(n_jobs=-1),
            'Lasso': Lasso(random_state=RANDOM_STATE),
            'Ridge': Ridge(random_state=RANDOM_STATE)
        }
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
    
    def load_data(self, filepath):
        """Load and prepare the dataset."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Basic validation
        required_columns = ['brand', 'model', 'year', 'kilometers', 'fuel_type', 
                          'gear_type', 'car_condition', 'price']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training."""
        logger.info("Preprocessing data...")
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Feature engineering
        current_year = datetime.now().year
        df['car_age'] = current_year - df['year']
        df['km_per_year'] = df['kilometers'] / (df['car_age'] + 1)  # +1 to avoid division by zero
        
        # Encode categorical variables
        categorical_cols = ['brand', 'model', 'fuel_type', 'gear_type', 'car_condition']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Define features and target
        X = df.drop('price', axis=1)
        y = df['price']
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and evaluate them."""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_test, y_pred)
                metrics['training_time'] = time.time() - start_time
                
                results[name] = metrics
                
                logger.info(f"{name} - R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:,.0f} SAR")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
        
        return results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,  # as percentage
            'r2': r2_score(y_true, y_pred),
            'mean_actual': np.mean(y_true),
            'mean_predicted': np.mean(y_pred)
        }
    
    def select_best_model(self, results):
        """Select the best model based on R² score."""
        best_r2 = -np.inf
        
        for name, metrics in results.items():
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                self.best_model_name = name
                self.best_model = self.models[name]
        
        logger.info(f"Best model: {self.best_model_name} (R²: {best_r2:.4f})")
        return self.best_model_name, best_r2
    
    def save_model_and_artifacts(self, results):
        """Save the best model and other artifacts."""
        if not self.best_model:
            raise ValueError("No model has been trained yet.")
        
        # Save the best model
        model_path = os.path.join(MODEL_DIR, 'car_price_model.pkl')
        joblib.dump(self.best_model, model_path)
        
        # Save label encoders
        for col, le in self.label_encoders.items():
            le_path = os.path.join(MODEL_DIR, f'le_{col}.pkl')
            joblib.dump(le, le_path)
        
        # Save feature names
        feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
        joblib.dump(self.feature_names, feature_names_path)
        
        # Save training results
        results_path = os.path.join(REPORTS_DIR, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_model': self.best_model_name,
                'metrics': results[self.best_model_name],
                'feature_importances': self.get_feature_importances(),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model and artifacts saved to {MODEL_DIR}")
    
    def get_feature_importances(self):
        """Get feature importances if available."""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(self.best_model, 'coef_'):
            coef = self.best_model.coef_
            return dict(zip(self.feature_names, coef))
        return {}

def main():
    """Main function to run the training pipeline."""
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Load and preprocess data
        data_path = os.path.join(DATA_DIR, 'cleaned_saudi_cars.csv')
        if not os.path.exists(data_path):
            logger.error(f"Cleaned data not found at {data_path}. Please run data_cleaning.py first.")
            sys.exit(1)
        
        df = trainer.load_data(data_path)
        X_train, X_test, y_train, y_test = trainer.preprocess_data(df)
        
        # Train models and evaluate
        results = trainer.train_models(X_train, X_test, y_train, y_test)
        
        # Select and save the best model
        best_model, best_score = trainer.select_best_model(results)
        trainer.save_model_and_artifacts(results)
        
        logger.info(f"\n🎉 Training completed successfully! Best model: {best_model} (R²: {best_score:.4f})")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
