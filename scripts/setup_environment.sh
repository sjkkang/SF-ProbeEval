#!/bin/bash
"""
Environment Setup Script for SF-ProbeEval

This script sets up the Python environment and installs all required dependencies
for the SF-ProbeEval pipeline.

Usage:
    ./scripts/setup_environment.sh [--dev]
    
Options:
    --dev    Install development dependencies (testing, linting, etc.)
"""

set -e  # Exit on any error

DEV_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🔧 Setting up SF-ProbeEval environment..."

# Check Python version
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install -r requirements.txt

# Install development dependencies if requested
if [ "$DEV_MODE" = true ]; then
    echo "🔬 Installing development dependencies..."
    pip install -e ".[dev]"
fi

# Download required NLTK data
echo "📥 Downloading NLTK data..."
python -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
print('NLTK data downloaded successfully')
"

# Download spaCy model
echo "🧠 Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📝 To activate the environment:"
echo "    source venv/bin/activate"
echo ""
echo "🚀 To run the full pipeline:"
echo "    ./scripts/run_full_pipeline.sh"
echo ""
echo "📊 To create expert annotations:"
echo "    python -m src.evaluation.create_expert_annotations --help"
