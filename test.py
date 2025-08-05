import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the predictor class
from inference import CarPricePredictor

# Constants
MODEL_DIR = 'models'
DATA_DIR = 'data/processed'
REPORTS_DIR = 'reports'
TEST_DATA_RATIO = 0.2  # 20% of data for testing
RANDOM_STATE = 42

# Create necessary directories
for directory in [MODEL_DIR, DATA_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class TestCarPriceModel(unittest.TestCase):
    """Test cases for the car price prediction model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before any tests are run."""
        # Initialize predictor
        cls.predictor = CarPricePredictor()
        
        # Load test data
        data_path = os.path.join(DATA_DIR, 'cleaned_saudi_cars.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cleaned data not found at {data_path}. Run data_cleaning.py first.")
        
        cls.df = pd.read_csv(data_path)
        
        # Split into train and test sets (same split as in training)
        test_size = int(len(cls.df) * TEST_DATA_RATIO)
        cls.test_df = cls.df.sample(n=test_size, random_state=RANDOM_STATE)
    
    def test_01_model_loading(self):
        """Test if the model and encoders are loaded correctly."""
        self.assertIsNotNone(self.predictor.model, "Model not loaded")
        self.assertTrue(hasattr(self.predictor, 'label_encoders'), "Label encoders not loaded")
        self.assertGreater(len(self.predictor.label_encoders), 0, "No label encoders loaded")
    
    def test_02_prediction_shape(self):
        """Test that predictions have the correct shape."""
        test_sample = self.test_df.iloc[0:1].to_dict('records')[0]
        prediction = self.predictor.predict([test_sample])
        
        self.assertIsInstance(prediction, np.ndarray, "Prediction should be a numpy array")
        self.assertEqual(prediction.shape, (1,), "Prediction shape mismatch")
    
    def test_03_prediction_values(self):
        """Test that predictions are within a reasonable range."""
        # Get the min and max prices from training data for validation
        min_price = self.df['price'].min() * 0.5  # Allow 50% below min
        max_price = self.df['price'].max() * 1.5  # Allow 50% above max
        
        # Test on a few random samples
        for _ in range(5):
            sample = self.test_df.sample(1).to_dict('records')[0]
            prediction = self.predictor.predict([sample])[0]
            
            self.assertGreaterEqual(prediction, min_price, 
                                  f"Prediction {prediction} below minimum expected price {min_price}")
            self.assertLessEqual(prediction, max_price,
                               f"Prediction {prediction} above maximum expected price {max_price}")
    
    def test_04_feature_importance(self):
        """Test that the model has feature importances."""
        # Only test if the model supports feature importances
        model = self.predictor.model
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            importances = self.predictor.get_feature_importances()
            self.assertGreater(len(importances), 0, "No feature importances found")
            
            # Check that all importances are non-negative
            for feature, importance in importances.items():
                self.assertGreaterEqual(importance, 0, f"Negative importance for feature {feature}")
    
    def test_05_performance_metrics(self):
        """Test that the model meets minimum performance criteria."""
        # Use a subset of test data for faster testing
        test_subset = self.test_df.sample(min(100, len(self.test_df)), random_state=RANDOM_STATE)
        
        # Make predictions
        X_test = test_subset.drop('price', axis=1).to_dict('records')
        y_true = test_subset['price'].values
        y_pred = self.predictor.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Log metrics
        print(f"\nPerformance on test set (n={len(test_subset)}):")
        print(f"  MAE: {mae:,.0f} SAR")
        print(f"  R²: {r2:.4f}")
        
        # Check if metrics meet minimum thresholds
        self.assertGreaterEqual(r2, 0.7, f"R² score {r2:.4f} is below threshold 0.7")
        
        # MAE should be less than 20% of the mean price
        mean_price = self.df['price'].mean()
        self.assertLessEqual(mae, mean_price * 0.2, 
                           f"MAE {mae:,.0f} is more than 20% of mean price {mean_price:,.0f}")
    
    def test_06_edge_cases(self):
        """Test the model with edge cases."""
        # Minimum values
        min_values = {
            'brand': self.df['brand'].mode()[0],  # Most common brand
            'model': self.df['model'].mode()[0],  # Most common model
            'year': self.df['year'].min(),
            'kilometers': self.df['kilometers'].min(),
            'fuel_type': self.df['fuel_type'].mode()[0],
            'gear_type': self.df['gear_type'].mode()[0],
            'car_condition': self.df['car_condition'].min()
        }
        
        # Maximum values
        max_values = {
            'brand': self.df['brand'].mode()[0],
            'model': self.df['model'].mode()[0],
            'year': datetime.now().year,
            'kilometers': self.df['kilometers'].max(),
            'fuel_type': self.df['fuel_type'].mode()[0],
            'gear_type': self.df['gear_type'].mode()[0],
            'car_condition': self.df['car_condition'].max()
        }
        
        # Test predictions
        for case_name, case_data in [('min_values', min_values), ('max_values', max_values)]:
            with self.subTest(case=case_name):
                try:
                    prediction = self.predictor.predict([case_data])[0]
                    self.assertIsInstance(prediction, (int, float, np.number), 
                                        f"Prediction for {case_name} is not a number")
                    self.assertGreaterEqual(prediction, 0, 
                                         f"Negative prediction for {case_name}")
                except Exception as e:
                    self.fail(f"Prediction failed for {case_name}: {str(e)}")

def generate_test_report():
    """Generate a test report with the results."""
    import io
    from contextlib import redirect_stdout
    
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestCarPriceModel)
    
    # Run tests and capture output
    buffer = io.StringIO()
    test_runner = unittest.TextTestRunner(stream=buffer, verbosity=2)
    
    with redirect_stdout(buffer):
        print("=" * 80)
        print(f"SAUDI CAR PRICE PREDICTION - MODEL TEST REPORT")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # Run tests
        test_result = test_runner.run(test_suite)
    
    # Get the test results
    report = {
        'timestamp': datetime.now().isoformat(),
        'tests_run': test_result.testsRun,
        'failures': len(test_result.failures),
        'errors': len(test_result.errors),
        'skipped': len(test_result.skipped),
        'success': test_result.wasSuccessful(),
        'output': buffer.getvalue()
    }
    
    # Save the report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, 'test_report.json')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print a summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Run:    {report['tests_run']}")
    print(f"Failures:     {report['failures']}")
    print(f"Errors:       {report['errors']}")
    print(f"Skipped:      {report['skipped']}")
    print(f"Success:      {report['success']}")
    print("=" * 80)
    print(f"Full report saved to: {os.path.abspath(report_path)}")
    
    return report['success']

if __name__ == "__main__":
    # Run tests and exit with appropriate status code
    success = generate_test_report()
    sys.exit(0 if success else 1)
