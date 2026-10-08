#!/usr/bin/env bash
set -e

echo "=== Initializing GestureOS Environment ==="
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting GestureOS..."
python main.py
