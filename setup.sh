#!/bin/bash
set -e
set -x

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

cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
make install

cd /mount/src/nse-swing-trading-system  # Streamlit Cloud uses this as working dir

pip install --no-cache-dir -r <(grep -v "ta-lib" requirements.txt)
pip install --no-cache-dir ta-lib==0.4.0

mkdir -p signals
[ ! -f signals/entry_signals.json ] && echo "{}" > signals/entry_signals.json

chmod 644 signals/entry_signals.json || true
chmod 644 EQUITY_L.csv || true

echo "✅ Setup complete"
