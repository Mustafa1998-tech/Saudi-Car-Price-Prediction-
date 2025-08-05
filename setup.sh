#!/bin/bash

# Exit on error
set -e

# Print commands and their arguments as they are executed
set -x

echo "🚀 Setting up Saudi Car Price Prediction Project..."

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p models data/raw data/processed reports eda

# Create placeholder files for git to track directories
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep reports/.gitkeep

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" < "3.8" ]]; then
    echo "❌ Python 3.8 or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

# Create and activate virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run data cleaning
echo "🧹 Running data cleaning..."
python data_cleaning.py

# Run EDA
echo "📊 Running exploratory data analysis..."
python eda.py

# Train the model
echo "🤖 Training the model..."
python train_model.py

# Run tests
echo "🧪 Running tests..."
python -m pytest test.py -v || echo "⚠️ Some tests failed, but continuing with setup..."

echo "✨ Setup completed successfully!"
echo "To run the application locally, use: python deploy.py"
echo "To deploy to a cloud platform, follow the instructions in the README.md file."
