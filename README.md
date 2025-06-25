# NSE Swing Trading System

A comprehensive swing trading system for NSE stocks that analyzes technical indicators, fundamental data, and market sentiment to identify potential trading opportunities.

## Features

- Technical Analysis using pandas_ta
- Fundamental Analysis
- Pattern Recognition
- Risk Management
- Real-time Stock Analysis
- Interactive Dashboard
- CSV Export Functionality

## Project Structure

```
swing_trader/
├── app.py                 # Main Streamlit application
├── main.py               # Core trading system logic
├── swing_setup_extension.py  # Pattern recognition and setup detection
├── EQUITY_L.csv          # NSE stock list
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies
├── setup.sh             # Setup script
├── .streamlit/          # Streamlit configuration
│   ├── config.toml
│   └── secrets.toml
└── signals/             # Trading signals
    └── entry_signals.json
```

## Setup

1. Install Python 3.9.0 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment

The application is configured for deployment on Streamlit Cloud:

1. Main file path: `app.py`
2. Python version: 3.9.0
3. Build command: `chmod +x setup.sh && ./setup.sh`
4. Start command: `streamlit run app.py`

## Configuration

- Minimum score threshold: 0-100 (default: 60)
- Market conditions: bull, bear, neutral
- Batch size: 20 stocks per batch

## Dependencies

- streamlit==1.32.0
- pandas==2.2.0
- numpy==1.26.4
- yfinance==0.2.36
- pandas_ta==0.3.14b0
- And other requirements listed in requirements.txt

## License

MIT License

## Live Demo

[View the live application on Streamlit Cloud](https://your-app-name.streamlit.app)

## Usage

1. Set your desired minimum score threshold (0-100)
2. Select the current market condition (bull/bear/neutral)
3. Click "Run Analysis" to start the process
4. View real-time results and recommendations
5. Download detailed analysis as CSV

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is for educational purposes only. Always do your own research before making investment decisions. 