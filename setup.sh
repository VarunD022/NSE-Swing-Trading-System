#!/bin/bash

apt-get update

# Avoid ta-lib packages that cause failure
apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev

pip install --no-cache-dir -r requirements.txt

# Create required signal file
mkdir -p signals
if [ ! -f "signals/entry_signals.json" ]; then
    echo "{}" > signals/entry_signals.json
fi

chmod 644 signals/entry_signals.json
chmod 644 EQUITY_L.csv

echo "Setup complete."
