#!/bin/bash
set -e

echo "Setting up virtual environment for SONAR-OSS..."

if [ -d "venv" ]; then
    echo "Virtual environment already exists. Remove it first if you want to recreate."
    exit 1
fi

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

echo "Installing psdn-sonar with dev extras..."
pip install -e ".[dev]"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python -c 'import psdn_sonar; print(psdn_sonar.__version__)'"
