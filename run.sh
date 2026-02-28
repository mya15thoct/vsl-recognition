#!/bin/bash

echo "=============================="
echo "STEP 1: Data preparation"
echo "=============================="
python main.py data prepare

echo ""
echo "=============================="
echo "STEP 2: Training"
echo "=============================="
python main.py train
