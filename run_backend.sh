#!/bin/bash
cd backend
echo "PyTorch CNN Visualization Tool - Starting backend server..."
echo

echo "Installing required packages..."
pip install -r requirements.txt
echo

echo "Starting backend server... (Press Ctrl+C to quit)"
python main.py
