import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = 'models'

class CarPricePredictor:
    """A class to handle car price predictions using the trained model."""
    
    def __init__(self, model_dir=MODEL_DIR):
        """Initialize the predictor by loading the model and encoders."""
        self.model_dir = Path(model_dir)
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.loaded = False
        self.load_model_and_encoders()
    
    def load_model_and_encoders(self):
        """Load the trained model and label encoders."""
        try:
            # Load the model
            model_path = self.model_dir / 'car_price_model.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.model = joblib.load(model_path)
            
            # Load label encoders
            encoder_files = {
                'brand': 'le_brand.pkl',
                'model': 'le_model.pkl',
                'fuel_type': 'le_fuel_type.pkl',
                'gear_type': 'le_gear_type.pkl',
                'car_condition': 'le_car_condition.pkl'
            }
            
            for col, filename in encoder_files.items():
                encoder_path = self.model_dir / filename
                if encoder_path.exists():
                    self.label_encoders[col] = joblib.load(encoder_path)
                else:
                    logger.warning(f"Encoder {filename} not found at {encoder_path}")
            
            # Load feature names
            feature_names_path = self.model_dir / 'feature_names.pkl'
            if feature_names_path.exists():
                self.feature_names = joblib.load(feature_names_path)
            else:
                logger.warning(f"Feature names not found at {feature_names_path}")
            
            self.loaded = True
            logger.info("Model and encoders loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model or encoders: {e}")
            raise
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction."""
        if not self.loaded:
            raise RuntimeError("Model and encoders not loaded. Call load_model_and_encoders() first.")
        
        # Create a DataFrame from input data
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise ValueError("Input must be a dictionary, list of dictionaries, or pandas DataFrame")
        
        # Validate required columns
        required_columns = ['brand', 'model', 'year', 'kilometers', 'fuel_type', 
                          'gear_type', 'car_condition']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Feature engineering
        current_year = datetime.now().year
        df['car_age'] = current_year - df['year']
        df['km_per_year'] = df['kilometers'] / (df['car_age'] + 1)  # +1 to avoid division by zero
        
        # Encode categorical variables
        for col in ['brand', 'model', 'fuel_type', 'gear_type', 'car_condition']:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories by encoding them as -1
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else -1)
                df[col] = le.transform(df[col].astype(str))
        
        # Ensure all expected features are present
        if hasattr(self, 'feature_names'):
            for col in self.feature_names:
                if col not in df.columns and col != 'price':
                    logger.warning(f"Feature {col} not found in input data. Adding with default value 0.")
                    df[col] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
        
        return df
    
    def predict(self, input_data):
        """
        Predict car price for the given input data.
        
        Args:
            input_data: Dictionary, list of dictionaries, or pandas DataFrame containing car features.
                      Required keys: brand, model, year, kilometers, fuel_type, gear_type, car_condition
                      
        Returns:
            Predicted price(s) as a numpy array
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model_and_encoders() first.")
        
        # Preprocess input
        df = self.preprocess_input(input_data)
        
        # Make predictions
        predictions = self.model.predict(df)
        
        # Ensure predictions are non-negative
        predictions = np.maximum(0, predictions)
        
        return predictions
    
    def predict_with_confidence(self, input_data, n_iterations=100):
        """
        Predict car price with confidence intervals using bootstrapping.
        Only works with models that support the `estimators_` attribute.
        """
        if not hasattr(self.model, 'estimators_'):
            logger.warning("Model does not support confidence intervals. Returning single prediction.")
            return self.predict(input_data), None, None
        
        # Preprocess input
        df = self.preprocess_input(input_data)
        
        # Get predictions from all estimators
        all_predictions = np.array([
            estimator.predict(df) for estimator in self.model.estimators_
        ])
        
        # Calculate statistics
        mean_prediction = np.mean(all_predictions, axis=0)
        lower_bound = np.percentile(all_predictions, 2.5, axis=0)
        upper_bound = np.percentile(all_predictions, 97.5, axis=0)
        
        # Ensure predictions are non-negative
        mean_prediction = np.maximum(0, mean_prediction)
        lower_bound = np.maximum(0, lower_bound)
        upper_bound = np.maximum(0, upper_bound)
        
        return mean_prediction, lower_bound, upper_bound

def format_price(price):
    """Format price with thousands separator and SAR symbol."""
    return f"{price:,.0f} ريال سعودي"

def main():
    """Example usage of the CarPricePredictor class."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict car prices using the trained model.')
    parser.add_argument('--brand', type=str, help='Car brand')
    parser.add_argument('--model', type=str, help='Car model')
    parser.add_argument('--year', type=int, help='Manufacturing year')
    parser.add_argument('--kilometers', type=int, help='Mileage in kilometers')
    parser.add_argument('--fuel_type', type=str, help='Fuel type (e.g., بنزين, ديزل)')
    parser.add_argument('--gear_type', type=str, help='Gear type (e.g., عادي, أوتوماتيك)')
    parser.add_argument('--car_condition', type=str, help='Car condition (e.g., جيدة, ممتازة)')
    parser.add_argument('--file', type=str, help='CSV file containing multiple car records')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = CarPricePredictor()
        
        if args.file:
            # Predict from CSV file
            df = pd.read_csv(args.file)
            predictions = predictor.predict(df)
            
            # Add predictions to the DataFrame
            df['predicted_price'] = predictions
            
            # Save results
            output_file = f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ Predictions saved to {output_file}")
            print("\nSample predictions:")
            print(df.head().to_string())
            
        else:
            # Predict single car
            if not all([args.brand, args.model, args.year, args.kilometers, 
                       args.fuel_type, args.gear_type, args.car_condition]):
                print("Error: Missing required arguments. Use --help for usage information.")
                return
            
            car_data = {
                'brand': args.brand,
                'model': args.model,
                'year': args.year,
                'kilometers': args.kilometers,
                'fuel_type': args.fuel_type,
                'gear_type': args.gear_type,
                'car_condition': args.car_condition
            }
            
            # Get prediction with confidence interval if available
            if hasattr(predictor.model, 'estimators_'):
                pred, lower, upper = predictor.predict_with_confidence(car_data)
                pred = pred[0]
                lower = lower[0]
                upper = upper[0]
                
                print("\n🚗 Car Details:")
                for key, value in car_data.items():
                    print(f"   {key}: {value}")
                
                print("\n💵 Predicted Price:")
                print(f"   {format_price(pred)}")
                print(f"   (95% Confidence Interval: {format_price(lower)} - {format_price(upper)})")
            else:
                pred = predictor.predict(car_data)[0]
                
                print("\n🚗 Car Details:")
                for key, value in car_data.items():
                    print(f"   {key}: {value}")
                
                print("\n💵 Predicted Price:")
                print(f"   {format_price(pred)}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
