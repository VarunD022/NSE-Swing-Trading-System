#!/bin/bash

# Create necessary directories
mkdir -p signals

# Install Python dependencies
pip install -r requirements.txt

# Create empty entry_signals.json if it doesn't exist
if [ ! -f "signals/entry_signals.json" ]; then
    echo "{}" > signals/entry_signals.json
fi

# Set permissions
chmod 644 signals/entry_signals.json
chmod 644 EQUITY_L.csv

echo "Setup completed successfully!" 