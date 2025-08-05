import os
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

def load_model():
    """Load the trained model with error handling."""
    try:
        # Try different possible model paths
        possible_paths = [
            os.path.join('models', 'car_price_model.pkl'),  # models/car_price_model.pkl
            'car_price_model.pkl',  # Current directory
            os.path.join(os.path.dirname(__file__), 'models', 'car_price_model.pkl'),  # models subfolder
            os.path.join(os.path.dirname(__file__), 'car_price_model.pkl')  # Same directory as script
        ]
        
        model = None
        model_path = None
        
        # Try each path until we find the model
        for path in possible_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                model_path = path
                print(f"Model loaded successfully from: {os.path.abspath(path)}")
                break
        
        if model is None:
            error_msg = """
            ❌ Error: Model file not found. Please ensure you have the model file in one of these locations:
            - models/car_price_model.pkl
            - car_price_model.pkl (in the same directory as app.py)
            
            If you don't have the model file, you need to train the model first by running:
            python train_model.py
            """
            st.error(error_msg)
            print(error_msg)
            return None
            
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        print(f"Error loading model: {str(e)}")
        return None

def load_encoders():
    """Load the label encoders with error handling."""
    try:
        # Define the encoder files we need to load
        encoder_files = {
            'brand': 'le_brand.pkl',
            'model': 'le_model.pkl',
            'fuel_type': 'le_fuel_type.pkl',
            'gear_type': 'le_gear_type.pkl',
            'car_condition': 'le_car_condition.pkl'
        }
        
        encoders = {}
        base_paths = [
            'models',  # models/le_*.pkl
            '.',       # le_*.pkl in current directory
            os.path.dirname(__file__)  # le_*.pkl in script directory
        ]
        
        # Try to find and load each encoder
        for encoder_name, encoder_file in encoder_files.items():
            encoder_loaded = False
            
            # Try each possible base path
            for base_path in base_paths:
                encoder_path = os.path.join(base_path, encoder_file)
                
                if os.path.exists(encoder_path):
                    try:
                        encoders[encoder_name] = joblib.load(encoder_path)
                        print(f"Loaded encoder for {encoder_name} from: {os.path.abspath(encoder_path)}")
                        encoder_loaded = True
                        break
                    except Exception as e:
                        print(f"Error loading {encoder_name} from {encoder_path}: {str(e)}")
            
            if not encoder_loaded:
                error_msg = f"❌ Error: Could not find or load encoder for {encoder_name}. " \
                           f"Please ensure you have the {encoder_file} file in the models directory."
                st.error(error_msg)
                print(error_msg)
                return None
        
        return encoders
        
    except Exception as e:
        st.error(f"Error loading encoders: {str(e)}")
        print(f"Error loading encoders: {str(e)}")
        return None

# Load model and encoders at the start
model = load_model()
encoders = load_encoders()

# Only show the prediction interface if the model loaded successfully
if model is not None and encoders is not None:
    # إعداد تصميم الصفحة
    st.set_page_config(page_title="توقع سعر السيارة", page_icon="🚗", layout="centered")

    # ترويسة جميلة
    st.markdown(
        """
        <h1 style='text-align: center; color: #2E86C1;'>🚗 توقع سعر السيارة المستعملة</h1>
        <p style='text-align: center; color: #566573;'>أدخل تفاصيل السيارة للحصول على السعر المتوقع بناءً على النموذج الذكي.</p>
        <hr style='border-top: 1px solid #ABB2B9;'>
        """,
        unsafe_allow_html=True
    )

    def show_prediction_interface():
        st.title('🚗 توقع سعر السيارة')
        st.write('أدخل تفاصيل السيارة للحصول على توقع السعر')
        
        with st.form("prediction_form"):
            # Input fields
            col1, col2 = st.columns(2)
            
            with col1:
                brand = st.selectbox('العلامة التجارية', ['Toyota', 'Hyundai', 'Kia', 'Ford', 'Chevrolet', 'BMW', 'Mercedes', 'Audi', 'Nissan'])
                model_name = st.text_input("الموديل", "A4")
                year = st.number_input("سنة الصنع", min_value=1990, max_value=2025, value=2020)
                fuel_type = st.selectbox('نوع الوقود', ['بنزين', 'ديزل', 'كهرباء', 'هايبرد'])
                
            with col2:
                mileage = st.number_input('عدد الكيلومترات', min_value=0, value=0)
                transmission = st.selectbox('ناقل الحركة', ['أوتوماتيك', 'عادي'])
                condition = st.selectbox('حالة السيارة', ['ممتازة', 'جيدة جدًا', 'جيدة', 'متوسطة', 'سيئة'])
            
            submit_button = st.form_submit_button("توقع السعر")
        
        if submit_button:
            with st.spinner('جاري حساب التوقع...'):
                try:
                    # Prepare input data
                    input_data = {
                        'brand': str(brand).strip().lower(),
                        'model': str(model_name).strip().lower(),
                        'year': int(year),
                        'fuel_type': str(fuel_type).strip().lower(),
                        'kilometers': float(mileage),
                        'gear_type': str(transmission).strip().lower(),
                        'car_condition': str(condition).strip().lower()
                    }
                    
                    # Make prediction
                    predicted_price = predict_price(input_data, model, encoders)
                    
                    if predicted_price is not None:
                        st.success(f"💰 السعر المتوقع: {predicted_price:,.2f} ريال سعودي")
                        
                except Exception as e:
                    st.error(f"حدث خطأ أثناء حساب التوقع: {str(e)}")
                    st.error("الرجاء التأكد من صحة البيانات المدخلة والمحاولة مرة أخرى.")

    def predict_price(input_data, model, encoders):
        """Make a prediction using the trained model."""
        try:
            # Debug: Print input data
            print("\n=== Input Data ===")
            print(input_data)
            
            # Define expected columns in the correct order based on model training
            expected_columns = [
                'brand', 'model', 'year', 'fuel_type', 'kilometers', 
                'gear_type', 'car_condition', 'age', 'km_per_year'
            ]
            
            # Create a DataFrame with the input data
            input_df = pd.DataFrame([{
                'brand': str(input_data['brand']).strip().lower(),
                'model': str(input_data['model']).strip().lower(),
                'year': int(input_data['year']),
                'fuel_type': str(input_data['fuel_type']).strip().lower(),
                'kilometers': float(input_data['kilometers']),
                'gear_type': str(input_data['gear_type']).strip().lower(),
                'car_condition': str(input_data['car_condition']).strip().lower()
            }])
            
            # Add engineered features
            current_year = datetime.now().year
            input_df['age'] = current_year - input_df['year']
            input_df['km_per_year'] = input_df['kilometers'] / (input_df['age'] + 1)
            
            # Ensure we have all expected columns
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0  # or appropriate default value
            
            # Reorder columns to match training data
            input_df = input_df[expected_columns]
            
            # Debug: Print final DataFrame
            print("\n=== Processed DataFrame ===")
            print(f"Columns: {input_df.columns.tolist()}")
            print(f"Shape: {input_df.shape}")
            print("First row values:", input_df.iloc[0].to_dict())
            
            # Encode categorical features
            for col in ['brand', 'model', 'fuel_type', 'gear_type', 'car_condition']:
                le = encoders[col]
                # Handle unseen labels by mapping to the most common class
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
                )
            
            # Make prediction
            prediction = model.predict(input_df)
            return float(prediction[0])
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"Error making prediction: {e}")
            print(f"\n=== Error Details ===\n{error_details}")
            print(f"\nInput data: {input_data}")
            if 'input_df' in locals():
                print(f"\nInput DataFrame columns: {input_df.columns.tolist()}")
                print(f"Input DataFrame shape: {input_df.shape}")
            return None

    show_prediction_interface()
else:
    st.error("❌ Unable to start the application. Please check the error messages above for details.")
    st.info("""💡 Make sure you have the following files in the correct location:
    - car_price_model.pkl
    - le_brand.pkl
    - le_model.pkl
    - le_fuel_type.pkl
    - le_gear_type.pkl
    - le_car_condition.pkl
    
    These files should be in the 'models' directory or the same directory as app.py""")
