#!/bin/bash
cd frontend
echo "PyTorch CNN Visualization Tool - Starting frontend server..."
echo

echo "Installing required packages..."
npm install
echo

echo "Starting frontend server... (Press Ctrl+C to quit)"
npm start
