import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Calculate Average True Range using pandas"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

def detect_swing_setups(df: pd.DataFrame) -> dict:
    """
    Detect various swing trading setups in the price data
    """
    setups = {
        "double_bottom": False,
        "double_top": False,
        "bull_flag": False,
        "bear_flag": False,
        "cup_and_handle": False
    }
    
    if df is None or len(df) < 50:
        return setups

    try:
        # Calculate ATR for volatility measurement
        atr = calculate_atr(df)
        
        # Double Bottom Pattern
        if len(df) >= 20:
            last_20 = df.tail(20)
            lows = last_20["Low"].values
            if len(lows) >= 20:
                # Find two similar lows within 20% of each other
                for i in range(len(lows)-10):
                    for j in range(i+10, len(lows)):
                        if abs(lows[i] - lows[j]) / lows[i] < 0.02:  # 2% threshold
                            setups["double_bottom"] = True
                            break

        # Double Top Pattern
        if len(df) >= 20:
            last_20 = df.tail(20)
            highs = last_20["High"].values
            if len(highs) >= 20:
                # Find two similar highs within 20% of each other
                for i in range(len(highs)-10):
                    for j in range(i+10, len(highs)):
                        if abs(highs[i] - highs[j]) / highs[i] < 0.02:  # 2% threshold
                            setups["double_top"] = True
                            break

        # Bull Flag Pattern
        if len(df) >= 30:
            # Look for strong uptrend followed by consolidation
            price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-30]) / df["Close"].iloc[-30]
            if price_change > 0.1:  # 10% uptrend
                recent_volatility = atr.iloc[-5:].mean() / df["Close"].iloc[-1]
                if recent_volatility < 0.02:  # Low volatility consolidation
                    setups["bull_flag"] = True

        # Bear Flag Pattern
        if len(df) >= 30:
            # Look for strong downtrend followed by consolidation
            price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-30]) / df["Close"].iloc[-30]
            if price_change < -0.1:  # 10% downtrend
                recent_volatility = atr.iloc[-5:].mean() / df["Close"].iloc[-1]
                if recent_volatility < 0.02:  # Low volatility consolidation
                    setups["bear_flag"] = True

        # Cup and Handle Pattern
        if len(df) >= 50:
            # Look for U-shaped pattern followed by small pullback
            last_50 = df.tail(50)
            mid_point = len(last_50) // 2
            
            # Check for U-shape in first half
            first_half = last_50.iloc[:mid_point]
            if (first_half["Low"].iloc[0] - first_half["Low"].min()) / first_half["Low"].iloc[0] > 0.1:
                # Check for handle in second half
                second_half = last_50.iloc[mid_point:]
                if (second_half["High"].max() - second_half["Close"].iloc[-1]) / second_half["High"].max() < 0.05:
                    setups["cup_and_handle"] = True

        return setups

    except Exception as e:
        print(f"Error detecting swing setups: {e}")
        return setups

def generate_entry_exit_levels(df: pd.DataFrame) -> dict:
    """
    Generate entry and exit levels based on technical analysis
    """
    levels = {
        "entry": 0,
        "stop_loss": 0,
        "target_1": 0,
        "target_2": 0
    }
    
    if df is None or len(df) < 20:
        return levels

    try:
        current_price = df["Close"].iloc[-1]
        atr = calculate_atr(df)
        
        # Calculate support and resistance levels
        support = df["Low"].tail(20).min()
        resistance = df["High"].tail(20).max()
        
        # Set entry price
        levels["entry"] = current_price
        
        # Set stop loss (2 ATR below entry)
        levels["stop_loss"] = current_price - (2 * atr.iloc[-1])
        
        # Set targets (2 and 3 ATR above entry)
        levels["target_1"] = current_price + (2 * atr.iloc[-1])
        levels["target_2"] = current_price + (3 * atr.iloc[-1])
        
        return levels

    except Exception as e:
        print(f"Error generating entry/exit levels: {e}")
        return levels

def log_entry_signal(symbol: str, setup_type: str, levels: dict):
    """
    Log entry signal to a JSON file
    """
    try:
        signal = {
            "symbol": symbol,
            "setup_type": setup_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "levels": levels
        }
        
        # Create signals directory if it doesn't exist
        signals_dir = "signals"
        if not os.path.exists(signals_dir):
            os.makedirs(signals_dir)
        
        # Append to signals file
        signals_file = os.path.join(signals_dir, "entry_signals.json")
        if os.path.exists(signals_file):
            with open(signals_file, "r") as f:
                signals = json.load(f)
        else:
            signals = []
        
        signals.append(signal)
        
        with open(signals_file, "w") as f:
            json.dump(signals, f, indent=2)
            
    except Exception as e:
        print(f"Error logging entry signal: {e}") 