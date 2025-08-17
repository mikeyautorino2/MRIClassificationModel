#!/bin/bash
# Quick setup and test script

echo "🚀 MRI Classification Pipeline Setup & Test"
echo "============================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# Create virtual environment (optional but recommended)
echo "🔧 Setting up virtual environment..."
python3 -m venv venv 2>/dev/null || echo "⚠️  Virtual environment creation skipped"

# Try to activate if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy pillow opencv-python scikit-learn matplotlib seaborn tqdm

# Install remaining dependencies
echo "📦 Installing remaining dependencies..."
pip install --quiet -r requirements.txt 2>/dev/null || echo "⚠️  Some packages may have failed to install"

# Run basic tests
echo "🧪 Running smoke tests..."
python3 test_basic.py

echo ""
echo "🎯 Next steps:"
echo "1. If tests pass, try: python train.py --config configs/base_config.yaml --debug"
echo "2. If tests fail, check error messages and install missing packages"
echo "3. For GPU support, reinstall PyTorch with CUDA"

echo ""
echo "📊 Quick training test (small dataset):"
echo "python train.py --config configs/base_config.yaml --debug"