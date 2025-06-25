#!/bin/bash

# Update package lists
apt-get update

# Install system dependencies (TA-Lib dependencies included)
apt-get install -y python3.9 python3.9-dev python3-pip build-essential \
    libssl-dev libffi-dev python3-setuptools libxml2-dev \
    libxslt1-dev zlib1g-dev wget tar

# Download and build TA-Lib from source
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install

# Return to app directory
cd /mount/src/nse-swing-trading-system

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Create necessary directories and files
mkdir -p signals
if [ ! -f "signals/entry_signals.json" ]; then
    echo "{}" > signals/entry_signals.json
fi

chmod 644 signals/entry_signals.json
chmod 644 EQUITY_L.csv

echo "✅ TA-Lib installed and setup completed!"
