#!/bin/bash

# Setup script for Sign Language Detection on Linux
# This script creates a conda environment and sets up the project

echo "========================================"
echo "Sign Language Detection - Linux Setup"
echo "========================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[OK] Conda found"

# Create conda environment
echo ""
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create conda environment"
    exit 1
fi

echo "[OK] Conda environment created successfully"

# Activate instructions
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate sign_language_detection"
echo ""
echo "Then you can use the project:"
echo "  1. Prepare dataset (optional):"
echo "     cd data && python prepare_dataset.py"
echo ""
echo "  2. Collect keypoints:"
echo "     cd sign_language_detection && python -m data.collect_data"
echo ""
echo "  3. Train model:"
echo "     cd sign_language_detection && python train.py"
echo ""
echo "  4. Run inference:"
echo "     cd sign_language_detection && python inference.py"
echo ""
