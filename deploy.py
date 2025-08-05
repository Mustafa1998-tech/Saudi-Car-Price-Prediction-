import os
import sys
import subprocess
import webbrowser
import time
import socket
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from the given port."""
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    return start_port  # Fallback to default if no port is available

def run_streamlit_app(port=8501):
    """Run the Streamlit app with the specified port."""
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    app_path = script_dir / 'app.py'
    
    if not app_path.exists():
        print(f"❌ Error: {app_path} not found!")
        return False
    
    # Find an available port
    port = find_available_port(port)
    
    # Build the command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "--server.port", str(port),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        str(app_path)
    ]
    
    # Print status
    print("🚀 Starting Saudi Car Price Prediction App...")
    print(f"🌐 Opening http://localhost:{port}")
    
    # Open the browser after a short delay
    def open_browser():
        time.sleep(2)  # Give the server a moment to start
        webbrowser.open_new_tab(f"http://localhost:{port}")
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run the Streamlit app
    try:
        process = subprocess.Popen(cmd, cwd=script_dir)
        process.wait()
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running the app: {e}")
        return False
    
    return True

def main():
    """Main function to handle deployment."""
    print("🔍 Checking requirements...")
    missing = check_requirements()
    
    if missing:
        print("❌ Missing required packages. Please install them with:")
        print(f"   pip install {' '.join(missing)}")
        return
    
    # Check if model exists
    model_path = Path("models/car_price_model.pkl")
    if not model_path.exists():
        print("⚠️  Model not found. Training a new model...")
        try:
            import train_model
            print("✅ Model trained successfully!")
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return
    
    # Run the Streamlit app
    run_streamlit_app()

if __name__ == "__main__":
    main()
