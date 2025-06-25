#!/bin/bash
set -e
set -x

echo "🔧 Step 1: Installing OS dependencies..."
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

echo "📦 Step 2: Downloading and building TA-Lib..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install

echo "📂 Step 3: Switching to repo directory..."
cd /mount/src/nse-swing-trading-system

echo "📦 Step 4: Installing Python packages (excluding ta-lib)..."
pip install --no-cache-dir -r <(grep -v "ta-lib" requirements.txt)

echo "📦 Step 5: Installing Python ta-lib wrapper..."
pip install --no-cache-dir ta-lib==0.4.0

echo "📁 Step 6: Preparing signals directory and files..."
mkdir -p signals
if [ ! -f "signals/entry_signals.json" ]; then
    echo "{}" > signals/entry_signals.json
fi

chmod 644 signals/entry_signals.json || true
chmod 644 EQUITY_L.csv || true

echo "✅ All setup steps completed successfully!"
