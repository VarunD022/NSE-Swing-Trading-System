#!/bin/bash
set -e
set -x

# Install core build tools
apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3.9 \
    python3.9-dev \
    python3-pip \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    python3-setuptools \
    tar

# Build TA-Lib from source
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install

# Go back to app directory
cd /mount/src/nse-swing-trading-system

# Install all Python dependencies *except* ta-lib
pip install --no-cache-dir -r <(grep -v "ta-lib" requirements.txt)

# Install ta-lib Python wrapper separately
pip install --no-cache-dir ta-lib==0.4.0

# Prepare app files
mkdir -p signals
if [ ! -f "signals/entry_signals.json" ]; then
    echo "{}" > signals/entry_signals.json
fi

chmod 644 signals/entry_signals.json || true
chmod 644 EQUITY_L.csv || true

echo "✅ Setup finished successfully"
