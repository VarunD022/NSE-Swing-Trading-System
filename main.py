# NSE Swing Trading System - Complete Implementation
# Author: Trading System
# Date: 2025

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import warnings
import os
from datetime import datetime, timedelta
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import random
import concurrent.futures
from functools import lru_cache
import threading
from queue import Queue
from swing_setup_extension import detect_swing_setups, generate_entry_exit_levels, log_entry_signal

# Suppress warnings
warnings.filterwarnings("ignore")

def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_window).mean()
    return k, d

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    return -100 * ((highest_high - close) / (highest_high - lowest_low))

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Commodity Channel Index"""
    tp = (high + low + close) / 3
    tp_sma = tp.rolling(window=window).mean()
    tp_std = tp.rolling(window=window).std()
    return (tp - tp_sma) / (0.015 * tp_std)

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average Directional Index"""
    tr = calculate_atr(high, low, close, window)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=window).mean() / tr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=window).mean() / tr
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=window).mean()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

class NSESwingTradingSystem:
    def __init__(self, output_path: str = r"C:\Users\Admin\Downloads\swing_trader", max_workers: int = 5):
        """
        Initialize the NSE Swing Trading System

        Args:
            output_path: Path to save output files
            max_workers: Maximum number of parallel workers
        """
        self.output_path = output_path
        self.setup_logging()
        self.ensure_output_directory()
        self.max_workers = max_workers
        self.nifty_data = None
        self.results = []
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Load stock symbols from CSV file
        self.smallcap_100_stocks = self.load_stock_symbols()

    def load_stock_symbols(self) -> List[str]:
        """
        Load stock symbols from the NSE stock list CSV file
        
        Returns:
            List of stock symbols
        """
        # Use the current directory instead of output_path
        symbols_file = "EQUITY_L.csv"
            
        try:
            if not os.path.exists(symbols_file):
                self.logger.error(f"Symbols file not found at {symbols_file}")
                raise FileNotFoundError(f"Required file not found: {symbols_file}")
                
            # Read symbols from CSV
            df = pd.read_csv(symbols_file)
            
            # Print column names to help identify the correct column
            self.logger.info(f"Available columns in the file: {df.columns.tolist()}")
            
            # Check if 'SYMBOL' column exists
            if 'SYMBOL' not in df.columns:
                self.logger.error("CSV file must contain a 'SYMBOL' column")
                raise ValueError("CSV file must contain a 'SYMBOL' column")
                
            # Get unique symbols and remove any whitespace
            symbols = df['SYMBOL'].str.strip().unique().tolist()
            
            if not symbols:
                self.logger.error("No symbols found in CSV file")
                raise ValueError("No symbols found in CSV file")
                
            self.logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error loading symbols from CSV: {e}")
            raise

    def setup_logging(self):
        """Setup logging configuration"""
        import sys
        import codecs

        # Create a custom StreamHandler that handles Unicode
        class UnicodeStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.output_path, "swing_trading.log"),
                    encoding='utf-8'
                ),
                UnicodeStreamHandler(sys.stdout)
            ],
        )
        self.logger = logging.getLogger(__name__)

    def ensure_output_directory(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            self.logger.info(f"Created output directory: {self.output_path}")

    @lru_cache(maxsize=1000)
    def fetch_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance with caching

        Args:
            symbol: Stock symbol
            period: Data period

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = f"{symbol}.NS"
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None

            df.reset_index(inplace=True)
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_nifty_data(self, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch Nifty 50 data for relative strength calculation"""
        try:
            nifty = yf.Ticker("^NSEI")
            df = nifty.history(period=period)
            if df.empty:
                return None
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching Nifty data: {e}")
            return None

    @lru_cache(maxsize=1000)
    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Fetch fundamental data for a stock with caching

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental metrics
        """
        try:
            ticker = f"{symbol}.NS"
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamental_data = {
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
                "roa": info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else 0,
                "current_ratio": info.get("currentRatio", 0),
                "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0,
                "revenue_growth": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else 0,
                "earnings_growth": info.get("earningsGrowth", 0) * 100 if info.get("earningsGrowth") else 0,
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "book_value": info.get("bookValue", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            }

            return fundamental_data

        except Exception as e:
            self.logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return {}

    def calculate_fundamental_score(self, fundamental_data: Dict) -> float:
        """
        Calculate fundamental analysis score (0-100)

        Args:
            fundamental_data: Dictionary with fundamental metrics

        Returns:
            Fundamental score (0-100)
        """
        if not fundamental_data:
            return 0

        score = 0

        # P/E Ratio Score (0-20 points)
        pe = fundamental_data.get("pe_ratio", 0)
        if pe > 0:
            if 10 <= pe <= 20:
                score += 20
            elif 20 < pe <= 25:
                score += 15
            elif 5 <= pe < 10:
                score += 10
            elif pe > 25:
                score += 5

        # ROE Score (0-20 points)
        roe = fundamental_data.get("roe", 0)
        if roe > 20:
            score += 20
        elif roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        elif roe > 5:
            score += 5

        # Debt-to-Equity Score (0-15 points)
        de_ratio = fundamental_data.get("debt_to_equity", 0)
        if de_ratio == 0:
            score += 15
        elif 0 < de_ratio < 30:
            score += 15
        elif 30 <= de_ratio < 50:
            score += 10
        elif 50 <= de_ratio < 100:
            score += 5

        # Profit Margin Score (0-15 points)
        profit_margin = fundamental_data.get("profit_margin", 0)
        if profit_margin > 20:
            score += 15
        elif profit_margin > 15:
            score += 12
        elif profit_margin > 10:
            score += 10
        elif profit_margin > 5:
            score += 5

        # Revenue Growth Score (0-15 points)
        revenue_growth = fundamental_data.get("revenue_growth", 0)
        if revenue_growth > 20:
            score += 15
        elif revenue_growth > 15:
            score += 12
        elif revenue_growth > 10:
            score += 10
        elif revenue_growth > 0:
            score += 5

        # Current Ratio Score (0-10 points)
        current_ratio = fundamental_data.get("current_ratio", 0)
        if 1.5 <= current_ratio <= 3.0:
            score += 10
        elif 1.0 <= current_ratio < 1.5 or 3.0 < current_ratio <= 4.0:
            score += 5

        # Price to Book Score (0-5 points)
        pb_ratio = fundamental_data.get("price_to_book", 0)
        if 0 < pb_ratio <= 1.5:
            score += 5
        elif 1.5 < pb_ratio <= 3.0:
            score += 3

        return min(score, 100)

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using pandas_ta

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators
        """
        if df is None or df.empty:
            return df

        try:
            # Moving Averages
            df["SMA_20"] = calculate_sma(df["Close"], 20)
            df["SMA_50"] = calculate_sma(df["Close"], 50)
            df["EMA_12"] = calculate_ema(df["Close"], 12)
            df["EMA_26"] = calculate_ema(df["Close"], 26)

            # RSI
            df["RSI"] = calculate_rsi(df["Close"])

            # MACD
            macd, signal_line, histogram = calculate_macd(df["Close"])
            df["MACD"] = macd
            df["MACD_signal"] = signal_line
            df["MACD_hist"] = histogram

            # Bollinger Bands
            upper_band, sma, lower_band = calculate_bollinger_bands(df["Close"])
            df["BB_upper"] = upper_band
            df["BB_middle"] = sma
            df["BB_lower"] = lower_band

            # Volume indicators
            df["Volume_SMA"] = calculate_sma(df["Volume"], 20)
            df["Volume_ratio"] = df["Volume"] / df["Volume_SMA"]

            # ADX
            df["ADX"] = calculate_adx(df["High"], df["Low"], df["Close"])

            # Stochastic
            stoch_k, stoch_d = calculate_stochastic(df["High"], df["Low"], df["Close"])
            df["STOCH_K"] = stoch_k
            df["STOCH_D"] = stoch_d

            # Williams %R
            df["WILLR"] = calculate_williams_r(df["High"], df["Low"], df["Close"])

            # CCI
            df["CCI"] = calculate_cci(df["High"], df["Low"], df["Close"])

            # ATR
            df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"])

            return df

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df

    def calculate_technical_score(self, df: pd.DataFrame) -> float:
        """
        Calculate technical analysis score (0-100)

        Args:
            df: DataFrame with technical indicators

        Returns:
            Technical score (0-100)
        """
        if df is None or df.empty or len(df) < 50:
            return 0

        try:
            latest = df.iloc[-1]
            score = 0

            # Trend Score (0-25 points)
            if (
                latest["Close"] > latest["SMA_20"]
                and latest["SMA_20"] > latest["SMA_50"]
            ):
                score += 25
            elif latest["Close"] > latest["SMA_20"]:
                score += 15
            elif latest["Close"] > latest["SMA_50"]:
                score += 10

            # RSI Score (0-20 points)
            rsi = latest["RSI"]
            if pd.notna(rsi):
                if 40 <= rsi <= 60:
                    score += 20
                elif 30 <= rsi <= 70:
                    score += 15
                elif 20 <= rsi < 30 or 70 < rsi <= 80:
                    score += 10
                elif rsi < 20 or rsi > 80:
                    score += 5

            # MACD Score (0-15 points)
            if (
                pd.notna(latest["MACD"])
                and pd.notna(latest["MACD_signal"])
                and latest["MACD"] > latest["MACD_signal"]
                and latest["MACD_hist"] > 0
            ):
                score += 15
            elif (
                pd.notna(latest["MACD"])
                and pd.notna(latest["MACD_signal"])
                and latest["MACD"] > latest["MACD_signal"]
            ):
                score += 10

            # Volume Score (0-15 points)
            if pd.notna(latest["Volume_ratio"]):
                if latest["Volume_ratio"] > 1.5:
                    score += 15
                elif latest["Volume_ratio"] > 1.2:
                    score += 10
                elif latest["Volume_ratio"] > 1.0:
                    score += 5

            # ADX Score (0-10 points)
            if pd.notna(latest["ADX"]):
                if latest["ADX"] > 25:
                    score += 10
                elif latest["ADX"] > 20:
                    score += 5

            # Bollinger Bands Score (0-10 points)
            if pd.notna(latest["BB_upper"]) and pd.notna(latest["BB_lower"]):
                bb_position = (latest["Close"] - latest["BB_lower"]) / (
                    latest["BB_upper"] - latest["BB_lower"]
                )
                if 0.2 <= bb_position <= 0.8:
                    score += 10
                elif 0.1 <= bb_position < 0.2 or 0.8 < bb_position <= 0.9:
                    score += 5

            # Stochastic Score (0-5 points)
            if pd.notna(latest["STOCH_K"]) and pd.notna(latest["STOCH_D"]):
                if (
                    20 <= latest["STOCH_K"] <= 80
                    and latest["STOCH_K"] > latest["STOCH_D"]
                ):
                    score += 5

            return min(score, 100)

        except Exception as e:
            self.logger.error(f"Error calculating technical score: {e}")
            return 0

    def calculate_momentum_indicators(
        self, df: pd.DataFrame, nifty_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate momentum indicators

        Args:
            df: Stock DataFrame
            nifty_df: Nifty DataFrame for relative strength

        Returns:
            DataFrame with momentum indicators
        """
        if df is None or df.empty:
            return df

        try:
            # Price momentum
            df["Price_momentum_5"] = (df["Close"] / df["Close"].shift(5) - 1) * 100
            df["Price_momentum_10"] = (df["Close"] / df["Close"].shift(10) - 1) * 100
            df["Price_momentum_20"] = (df["Close"] / df["Close"].shift(20) - 1) * 100

            # Rate of Change
            df["ROC_10"] = df["Close"].pct_change(10)
            df["ROC_20"] = df["Close"].pct_change(20)

            # Relative strength vs Nifty
            if nifty_df is not None and not nifty_df.empty:
                # Align dates
                merged = pd.merge(
                    df[["Date", "Close"]],
                    nifty_df[["Date", "Close"]],
                    on="Date",
                    suffixes=("_stock", "_nifty"),
                    how="inner",
                )
                if not merged.empty:
                    stock_returns = merged["Close_stock"].pct_change(20)
                    nifty_returns = merged["Close_nifty"].pct_change(20)
                    relative_strength = (stock_returns - nifty_returns) * 100

                    # Map back to original dataframe
                    df = df.merge(
                        merged[["Date"]].assign(Relative_strength=relative_strength),
                        on="Date",
                        how="left",
                    )

            # Volume Price Trend
            df["VPT"] = (
                (
                    df["Volume"]
                    * (df["Close"] - df["Close"].shift())
                    / df["Close"].shift()
                )
                .fillna(0)
                .cumsum()
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return df

    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum score (0-100)

        Args:
            df: DataFrame with momentum indicators

        Returns:
            Momentum score (0-100)
        """
        if df is None or df.empty:
            return 0

        try:
            latest = df.iloc[-1]
            score = 0

            # Price momentum 20-day score (0-25 points)
            momentum_20 = latest.get("Price_momentum_20", 0)
            if pd.notna(momentum_20):
                if momentum_20 > 15:
                    score += 25
                elif momentum_20 > 10:
                    score += 20
                elif momentum_20 > 5:
                    score += 15
                elif momentum_20 > 0:
                    score += 10
                elif momentum_20 > -5:
                    score += 5

            # Price momentum 10-day score (0-20 points)
            momentum_10 = latest.get("Price_momentum_10", 0)
            if pd.notna(momentum_10):
                if momentum_10 > 10:
                    score += 20
                elif momentum_10 > 5:
                    score += 15
                elif momentum_10 > 0:
                    score += 10
                elif momentum_10 > -3:
                    score += 5

            # Relative strength score (0-25 points)
            rel_strength = latest.get("Relative_strength", 0)
            if pd.notna(rel_strength):
                if rel_strength > 10:
                    score += 25
                elif rel_strength > 5:
                    score += 20
                elif rel_strength > 0:
                    score += 15
                elif rel_strength > -5:
                    score += 10
                elif rel_strength > -10:
                    score += 5

            # ROC score (0-15 points)
            roc_20 = latest.get("ROC_20", 0)
            if pd.notna(roc_20):
                if roc_20 > 10:
                    score += 15
                elif roc_20 > 5:
                    score += 10
                elif roc_20 > 0:
                    score += 5

            # Short-term momentum (0-15 points)
            momentum_5 = latest.get("Price_momentum_5", 0)
            if pd.notna(momentum_5):
                if momentum_5 > 5:
                    score += 15
                elif momentum_5 > 2:
                    score += 10
                elif momentum_5 > 0:
                    score += 5

            return min(score, 100)

        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0

    def identify_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Identify candlestick patterns using pandas_ta

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with pattern scores
        """
        if df is None or df.empty or len(df) < 10:
            return {}

        try:
            patterns = {}

            # Bullish patterns
            patterns["Hammer"] = (df["Close"] - df["Open"] > 0) & (df["High"] - df["Close"] < 0.05) & (df["Low"] - df["Open"] < 0.05)
            patterns["Bullish_Engulfing"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Morning_Star"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Piercing"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Three_White_Soldiers"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Dragonfly_Doji"] = (df["Close"] - df["Open"] < 0.05) & (df["High"] - df["Close"] < 0.05) & (df["Low"] - df["Open"] < 0.05)
            patterns["Bullish_Harami"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Bullish_Belt_Hold"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Bullish_Kicker"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Bullish_Three_Line_Strike"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Bullish_Three_Inside_Up"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Bullish_Three_Outside_Up"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))

            # Bearish patterns
            patterns["Shooting_Star"] = (df["Close"] < df["Open"]) & (df["High"] - df["Close"] < 0.05) & (df["Low"] - df["Open"] < 0.05)
            patterns["Bearish_Engulfing"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Evening_Star"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Dark_Cloud"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Three_Black_Crows"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Gravestone_Doji"] = (df["Close"] - df["Open"] < 0.05) & (df["High"] - df["Close"] < 0.05) & (df["Low"] - df["Open"] < 0.05)
            patterns["Bearish_Harami"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Bearish_Belt_Hold"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Bearish_Kicker"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Bearish_Three_Line_Strike"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))
            patterns["Bearish_Three_Inside_Down"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) > df["Open"].shift(2))
            patterns["Bearish_Three_Outside_Down"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1)) & (df["Close"].shift(2) < df["Open"].shift(2))

            # Add volume confirmation for patterns
            for pattern_name in patterns:
                if len(patterns[pattern_name]) > 0:
                    pattern_dates = np.where(patterns[pattern_name] != 0)[0]
                    for date_idx in pattern_dates:
                        if date_idx > 0:
                            avg_volume = np.mean(df["Volume"].iloc[max(0, date_idx-20):date_idx])
                            if df["Volume"].iloc[date_idx] < avg_volume * 1.2:  # Volume should be 20% above average
                                patterns[pattern_name][date_idx] = 0

            return patterns

        except Exception as e:
            self.logger.error(f"Error identifying candlestick patterns: {e}")
            return {}

    def calculate_pattern_score(self, patterns: Dict) -> float:
        """
        Calculate enhanced candlestick pattern score (-20 to +20)

        Args:
            patterns: Dictionary with pattern arrays

        Returns:
            Pattern score (-20 to +20)
        """
        if not patterns:
            return 0

        try:
            bullish_patterns = [
                "Hammer", "Bullish_Engulfing", "Morning_Star", "Piercing",
                "Three_White_Soldiers", "Dragonfly_Doji", "Bullish_Harami",
                "Bullish_Belt_Hold", "Bullish_Kicker", "Bullish_Three_Line_Strike",
                "Bullish_Three_Inside_Up", "Bullish_Three_Outside_Up"
            ]
            
            bearish_patterns = [
                "Shooting_Star", "Bearish_Engulfing", "Evening_Star", "Dark_Cloud",
                "Three_Black_Crows", "Gravestone_Doji", "Bearish_Harami",
                "Bearish_Belt_Hold", "Bearish_Kicker", "Bearish_Three_Line_Strike",
                "Bearish_Three_Inside_Down", "Bearish_Three_Outside_Down"
            ]

            bullish_score = 0
            bearish_score = 0

            # Check last 5 candles for patterns
            for pattern_name, pattern_array in patterns.items():
                if len(pattern_array) > 0:
                    recent_signals = pattern_array[-5:]  # Last 5 candles

                    if pattern_name in bullish_patterns:
                        # Weight recent patterns more heavily
                        for i, signal in enumerate(recent_signals):
                            if signal > 0:
                                weight = 1.0 if i >= 3 else 0.8  # More weight to recent patterns
                                bullish_score += weight
                    elif pattern_name in bearish_patterns:
                        for i, signal in enumerate(recent_signals):
                            if signal < 0:
                                weight = 1.0 if i >= 3 else 0.8
                                bearish_score += weight

            # Calculate final score with pattern strength consideration
            if bullish_score > bearish_score:
                return min(bullish_score * 4, 20)  # Increased multiplier for stronger signals
            elif bearish_score > bullish_score:
                return max(-bearish_score * 4, -20)
            else:
                return 0

        except Exception as e:
            self.logger.error(f"Error calculating pattern score: {e}")
            return 0

    def calculate_pattern_strength(self, df, pattern_idx):
        candle_size = abs(df["Close"].iloc[pattern_idx] - df["Open"].iloc[pattern_idx])
        avg_candle_size = df["Close"].pct_change().abs().mean()
        return candle_size / avg_candle_size

    def fetch_stock_news(self, symbol: str) -> List[Dict]:
        """
        Fetch recent news for a stock with rate limiting and user-agent rotation.

        Args:
            symbol: Stock symbol

        Returns:
            List of news items
        """
        try:
            # Simple news fetching using Google News RSS
            url = f"https://news.google.com/rss/search?q={symbol}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"

            # User-Agent rotation
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
                "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
            ]
            headers = {"User-Agent": random.choice(user_agents)}

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.content, "xml")
            news_items = []

            for item in soup.find_all("item")[:5]:  # Get latest 5 news items
                title = item.title.text if item.title else ""
                pub_date = item.pubDate.text if item.pubDate else ""
                news_items.append({"title": title, "date": pub_date})

            # Rate limiting
            time.sleep(random.uniform(1, 3))  # Random delay between 1 to 3 seconds

            return news_items

        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def analyze_news_sentiment(self, news_items: List[Dict]) -> float:
        """
        Analyze news sentiment

        Args:
            news_items: List of news items

        Returns:
            Sentiment score (-15 to +15)
        """
        if not news_items:
            return 0

        try:
            total_sentiment = 0
            valid_items = 0

            for item in news_items:
                title = item.get("title", "")
                if title:
                    blob = TextBlob(title)
                    sentiment = blob.sentiment.polarity
                    total_sentiment += sentiment
                    valid_items += 1

            if valid_items == 0:
                return 0

            avg_sentiment = total_sentiment / valid_items

            # Scale sentiment to -15 to +15
            sentiment_score = avg_sentiment * 15

            return max(-15, min(15, sentiment_score))

        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return 0

    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics

        Args:
            df: DataFrame with price data

        Returns:
            Dictionary with risk metrics
        """
        if df is None or df.empty or len(df) < 20:
            return {}

        try:
            returns = df["Close"].pct_change().dropna()

            risk_metrics = {
                "volatility": returns.std()
                * np.sqrt(252)
                * 100,  # Annualized volatility
                "beta": 0,  # Will be calculated if Nifty data is available
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "var_95": np.percentile(returns, 5) * 100,  # 95% VaR
                "skewness": returns.skew(),
                "kurtosis": returns.kurtosis(),
            }

            # Calculate maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            risk_metrics["max_drawdown"] = drawdown.min() * 100

            # Calculate Sharpe ratio (assuming risk-free rate of 6%)
            excess_returns = returns - 0.06 / 252  # Daily risk-free rate
            if excess_returns.std() != 0:
                risk_metrics["sharpe_ratio"] = (
                    excess_returns.mean() / excess_returns.std()
                ) * np.sqrt(252)

            return risk_metrics

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def calculate_risk_score(self, risk_metrics: Dict) -> float:
        """
        Calculate risk score (0-100, higher is better/lower risk)

        Args:
            risk_metrics: Dictionary with risk metrics

        Returns:
            Risk score (0-100)
        """
        if not risk_metrics:
            return 50  # Neutral score

        try:
            score = 0

            # Volatility score (0-30 points)
            volatility = risk_metrics.get("volatility", 50)
            if volatility < 20:
                score += 30
            elif volatility < 30:
                score += 25
            elif volatility < 40:
                score += 20
            elif volatility < 50:
                score += 15
            elif volatility < 60:
                score += 10
            else:
                score += 5

            # Maximum drawdown score (0-25 points)
            max_dd = abs(risk_metrics.get("max_drawdown", 50))
            if max_dd < 10:
                score += 25
            elif max_dd < 15:
                score += 20
            elif max_dd < 20:
                score += 15
            elif max_dd < 30:
                score += 10
            elif max_dd < 40:
                score += 5

            # Sharpe ratio score (0-25 points)
            sharpe = risk_metrics.get("sharpe_ratio", 0)
            if sharpe > 2:
                score += 25
            elif sharpe > 1.5:
                score += 20
            elif sharpe > 1:
                score += 15
            elif sharpe > 0.5:
                score += 10
            elif sharpe > 0:
                score += 5

            # VaR score (0-20 points)
            var_95 = abs(risk_metrics.get("var_95", 5))
            if var_95 < 2:
                score += 20
            elif var_95 < 3:
                score += 15
            elif var_95 < 4:
                score += 10
            elif var_95 < 5:
                score += 5

            return min(score, 100)

        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 50

    def calculate_additional_technical_indicators(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate additional technical indicators like OBV and detect divergences.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional technical indicators
        """
        if df is None or df.empty:
            return df

        try:
            # On-Balance Volume (OBV)
            df["OBV"] = df["Close"].diff(1)
            df["OBV"] = np.where(df["OBV"] > 0, df["Volume"], np.where(df["OBV"] < 0, -df["Volume"], 0))

            # Detect divergence (example with RSI)
            df["RSI_Divergence"] = (
                df["RSI"] - df["Close"].pct_change().rolling(window=14).mean()
            )

            return df
        except Exception as e:
            self.logger.error(f"Error calculating additional technical indicators: {e}")
            return df

    def adjust_scoring_weights(
        self, market_condition: str
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Adjust scoring weights based on market conditions.

        Args:
            market_condition: Current market condition ('bull', 'bear', 'neutral')

        Returns:
            Tuple of adjusted weights for technical, momentum, fundamental, risk, pattern, and sentiment scores
        """
        if market_condition == "bull":
            return 0.2, 0.3, 0.2, 0.1, 0.1, 0.1
        elif market_condition == "bear":
            return 0.3, 0.2, 0.3, 0.1, 0.05, 0.05
        else:  # neutral
            return 0.25, 0.25, 0.25, 0.15, 0.05, 0.05

    def calculate_trailing_stop_loss(
        self, current_price: float, atr: float, trailing_multiplier: float = 2.0
    ) -> float:
        """
        Calculate trailing stop-loss based on ATR (Average True Range).

        Args:
            current_price: Current stock price
            atr: Average True Range value
            trailing_multiplier: Multiplier for ATR to set the stop-loss distance

        Returns:
            Trailing stop-loss price
        """
        return current_price - (trailing_multiplier * atr)

    def fetch_social_media_sentiment(self, symbol: str) -> float:
        """
        Fetch and analyze social media sentiment for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Sentiment score (-5 to +5)
        """
        try:
            # Placeholder for social media sentiment analysis
            # In a real implementation, you would use an API like Twitter API or a sentiment analysis library
            # Here, we simulate a sentiment score
            simulated_sentiment_score = 0.5  # Simulated score for demonstration
            return simulated_sentiment_score
        except Exception as e:
            self.logger.error(
                f"Error fetching social media sentiment for {symbol}: {e}"
            )
            return 0

    def analyze_stock(self, symbol: str, market_condition: str = "neutral") -> Dict:
        """
        Perform comprehensive analysis of a stock with enhanced filters

        Args:
            symbol: Stock symbol
            market_condition: Current market condition

        Returns:
            Dictionary with analysis results
        """
        try:
            # Fetch all data in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                stock_data_future = executor.submit(self.fetch_stock_data, symbol)
                fundamental_future = executor.submit(self.get_fundamental_data, symbol)
                news_future = executor.submit(self.fetch_stock_news, symbol)
                
                df = stock_data_future.result()
                fundamental_data = fundamental_future.result()
                news_items = news_future.result()

            if df is None or df.empty:
                return {"symbol": symbol, "error": "No data available"}

            # Price filter
            current_price = df["Close"].iloc[-1]
            if current_price < 50:  # Filter out stocks below Rs. 50
                return {"symbol": symbol, "error": "Price too low"}

            # Volume filter
            avg_volume = df["Volume"].mean()
            if avg_volume < 100000:  # Filter out low volume stocks
                return {"symbol": symbol, "error": "Insufficient volume"}

            # Calculate technical indicators first
            df = self.calculate_technical_indicators(df)
            df = self.calculate_additional_technical_indicators(df)
            df = self.calculate_momentum_indicators(df, self.nifty_data)

            # Now check trend confirmation after calculating indicators
            if "SMA_20" not in df.columns or "SMA_50" not in df.columns:
                return {"symbol": symbol, "error": "Failed to calculate moving averages"}

            sma_20 = df["SMA_20"].iloc[-1]
            sma_50 = df["SMA_50"].iloc[-1]
            if pd.isna(sma_20) or pd.isna(sma_50):
                return {"symbol": symbol, "error": "Invalid moving average values"}

            if current_price < sma_20 or sma_20 < sma_50:
                return {"symbol": symbol, "error": "Not in uptrend"}

            # Identify patterns and calculate scores
            patterns = self.identify_candlestick_patterns(df)
            news_sentiment_score = self.analyze_news_sentiment(news_items)
            social_media_sentiment_score = self.fetch_social_media_sentiment(symbol)
            risk_metrics = self.calculate_risk_metrics(df)

            # Calculate all scores
            technical_score = self.calculate_technical_score(df)
            momentum_score = self.calculate_momentum_score(df)
            fundamental_score = self.calculate_fundamental_score(fundamental_data)
            pattern_score = self.calculate_pattern_score(patterns)
            sentiment_score = (news_sentiment_score + social_media_sentiment_score) / 2
            risk_score = self.calculate_risk_score(risk_metrics)

            # Get current price and calculate key levels
            support_level = df["Low"].tail(20).min()
            resistance_level = df["High"].tail(20).max()

            # Calculate stop loss and target levels with ATR
            atr = df["ATR"].iloc[-1] if "ATR" in df.columns and pd.notna(df["ATR"].iloc[-1]) else current_price * 0.02
            stop_loss = current_price - (2 * atr)
            trailing_stop_loss = self.calculate_trailing_stop_loss(current_price, atr)
            target_1 = current_price + (2 * atr)
            target_2 = current_price + (3 * atr)

            # Trend confirmation
            trend_strength = 1.0
            if current_price < sma_20 or sma_20 < sma_50:
                trend_strength = 0.5  # Reduce score for stocks not in uptrend

            # Adjust scoring weights based on market condition
            tech_w, mom_w, fund_w, risk_w, pat_w, sent_w = self.adjust_scoring_weights(market_condition)

            # Calculate composite score with trend confirmation
            composite_score = (
                technical_score * tech_w
                + momentum_score * mom_w
                + fundamental_score * fund_w
                + risk_score * risk_w
                + (pattern_score + 20) * pat_w * 2.5
                + (sentiment_score + 5) * sent_w * 10
            ) * trend_strength

            # Detect and log swing setups
            setups = detect_swing_setups(df)
            for setup_name, triggered in setups.items():
                if triggered:
                    levels = generate_entry_exit_levels(df)
                    log_entry_signal(symbol, setup_name, levels)
                    break  # Log first detected setup only (optional)

            # Add entry signal for top stocks
            if composite_score >= 60:  # Only log signals for stocks with good scores
                entry_signal = {
                    "symbol": symbol,
                    "setup_type": "Composite Score",
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "target_1": target_1,
                    "target_2": target_2,
                    "score": composite_score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                log_entry_signal(symbol, "Composite Score", entry_signal)

            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "composite_score": round(composite_score, 2),
                "technical_score": round(technical_score, 2),
                "momentum_score": round(momentum_score, 2),
                "fundamental_score": round(fundamental_score, 2),
                "pattern_score": round(pattern_score, 2),
                "sentiment_score": round(sentiment_score, 2),
                "risk_score": round(risk_score, 2),
                "support_level": round(support_level, 2),
                "resistance_level": round(resistance_level, 2),
                "stop_loss": round(stop_loss, 2),
                "trailing_stop_loss": round(trailing_stop_loss, 2),
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "risk_reward_ratio": round((target_1 - current_price) / (current_price - stop_loss), 2),
                "volatility": round(risk_metrics.get("volatility", 0), 2),
                "max_drawdown": round(risk_metrics.get("max_drawdown", 0), 2),
                "volume_ratio": round(df["Volume_ratio"].iloc[-1], 2) if "Volume_ratio" in df.columns else 0,
                "rsi": round(df["RSI"].iloc[-1], 2) if "RSI" in df.columns and pd.notna(df["RSI"].iloc[-1]) else 0,
                "trend_strength": round(trend_strength, 2),
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def screen_stocks(self, min_score: float = 60) -> List[Dict]:
        """
        Screen stocks based on composite score with parallel processing

        Args:
            min_score: Minimum composite score threshold

        Returns:
            List of stocks meeting criteria
        """
        self.logger.info("Starting stock screening process...")

        # Fetch Nifty data for relative strength calculation
        self.nifty_data = self.fetch_nifty_data()

        # Split stocks into batches for parallel processing
        batch_size = 20  # Process 20 stocks at a time
        all_results = []
        
        for i in range(0, len(self.smallcap_100_stocks), batch_size):
            batch = self.smallcap_100_stocks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.smallcap_100_stocks) + batch_size - 1)//batch_size}")
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_symbol = {
                    executor.submit(self.analyze_stock, symbol): symbol 
                    for symbol in batch
                }
                
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result and "error" not in result:
                            all_results.append(result)
                            if result["composite_score"] >= min_score:
                                self.logger.info(f"+ {symbol}: Score {result['composite_score']}")
                    except Exception as e:
                        self.logger.error(f"Error analyzing {symbol}: {e}")
            
            # Add a small delay between batches to avoid rate limiting
            time.sleep(1)

        # Sort by composite score
        all_results.sort(key=lambda x: x["composite_score"], reverse=True)

        # Filter by minimum score
        filtered_results = [r for r in all_results if r["composite_score"] >= min_score]

        self.logger.info(
            f"Screening complete. Found {len(filtered_results)} stocks above score {min_score}"
        )

        return filtered_results

    def analyze_stock_batch(self, symbols: List[str], market_condition: str = "neutral") -> List[Dict]:
        """
        Analyze a batch of stocks in parallel

        Args:
            symbols: List of stock symbols
            market_condition: Current market condition

        Returns:
            List of analysis results
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_stock, symbol, market_condition): symbol 
                for symbol in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and "error" not in result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol}: {e}")
                
        return results

    def save_results(self, results: List[Dict], filename: str = None):
        """
        Save results to CSV and JSON files

        Args:
            results: List of analysis results
            filename: Optional custom filename
        """
        if not results:
            self.logger.warning("No results to save")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if filename is None:
                base_filename = f"nse_swing_trading_results_{timestamp}"
            else:
                base_filename = f"{filename}_{timestamp}"

            # Save to CSV
            df = pd.DataFrame(results)
            csv_path = os.path.join(self.output_path, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Results saved to: {csv_path}")

            # Save to JSON
            json_path = os.path.join(self.output_path, f"{base_filename}.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to: {json_path}")

            # Save entry signals for top 50 stocks
            top_50_results = sorted(results, key=lambda x: x["composite_score"], reverse=True)[:50]
            entry_signals = []
            for stock in top_50_results:
                entry_signal = {
                    "symbol": stock["symbol"],
                    "setup_type": "Composite Score",
                    "entry_price": stock["current_price"],
                    "stop_loss": stock["stop_loss"],
                    "target_1": stock["target_1"],
                    "target_2": stock["target_2"],
                    "score": stock["composite_score"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                entry_signals.append(entry_signal)

            # Save entry signals to CSV
            entry_signals_df = pd.DataFrame(entry_signals)
            entry_signals_path = os.path.join(self.output_path, f"entry_signals_{timestamp}.csv")
            entry_signals_df.to_csv(entry_signals_path, index=False)
            self.logger.info(f"Entry signals saved to: {entry_signals_path}")

            # Create summary report
            self.create_summary_report(results, base_filename)

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def create_summary_report(self, results: List[Dict], filename: str):
        """
        Create a summary report

        Args:
            results: List of analysis results
            filename: Base filename for the report
        """
        try:
            report_path = os.path.join(self.output_path, f"{filename}_report.txt")

            with open(report_path, "w", encoding='utf-8') as f:
                f.write("NSE SWING TRADING SYSTEM - ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Total Stocks Analyzed: {len(results)}\n\n")

                if results:
                    # Top 10 stocks
                    f.write("TOP 10 STOCKS BY COMPOSITE SCORE:\n")
                    f.write("-" * 40 + "\n")

                    for i, stock in enumerate(results[:10], 1):
                        f.write(
                            f"{i:2d}. {stock['symbol']:12s} - Score: {stock['composite_score']:6.2f} "
                            f"Price: Rs.{stock['current_price']:8.2f} R:R: {stock['risk_reward_ratio']:4.2f}\n"
                        )

                    f.write("\n\nDETAILED ANALYSIS (Top 5):\n")
                    f.write("=" * 50 + "\n")

                    for i, stock in enumerate(results[:5], 1):
                        f.write(
                            f"\n{i}. {stock['symbol']} - Overall Score: {stock['composite_score']:.2f}\n"
                        )
                        f.write(f"   Current Price: Rs.{stock['current_price']:.2f}\n")
                        f.write(f"   Technical Score: {stock['technical_score']:.2f}\n")
                        f.write(f"   Momentum Score: {stock['momentum_score']:.2f}\n")
                        f.write(
                            f"   Fundamental Score: {stock['fundamental_score']:.2f}\n"
                        )
                        f.write(f"   Risk Score: {stock['risk_score']:.2f}\n")
                        f.write(f"   Support: Rs.{stock['support_level']:.2f}\n")
                        f.write(f"   Resistance: Rs.{stock['resistance_level']:.2f}\n")
                        f.write(f"   Stop Loss: Rs.{stock['stop_loss']:.2f}\n")
                        f.write(f"   Target 1: Rs.{stock['target_1']:.2f}\n")
                        f.write(f"   Target 2: Rs.{stock['target_2']:.2f}\n")
                        f.write(f"   Risk:Reward: 1:{stock['risk_reward_ratio']:.2f}\n")
                        f.write(f"   Volatility: {stock['volatility']:.2f}%\n")
                        f.write("-" * 50 + "\n")

                f.write("\n\nDISCLAIMER:\n")
                f.write("This analysis is for educational purposes only.\n")
                f.write(
                    "Please consult with a financial advisor before making investment decisions.\n"
                )
                f.write("Past performance does not guarantee future results.\n")

            self.logger.info(f"Summary report saved to: {report_path}")

        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")

    def create_visualization(self, results: List[Dict], filename: str = None):
        """
        Create visualization charts

        Args:
            results: List of analysis results
            filename: Optional custom filename
        """
        if not results:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if filename is None:
                base_filename = f"nse_swing_trading_charts_{timestamp}"
            else:
                base_filename = f"{filename}_charts_{timestamp}"

            # Set up the plotting style
            plt.style.use("seaborn-v0_8")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(
                "NSE Swing Trading System - Analysis Results",
                fontsize=16,
                fontweight="bold",
            )

            # Top 10 stocks by composite score
            top_10 = results[:10]
            symbols = [r["symbol"] for r in top_10]
            scores = [r["composite_score"] for r in top_10]

            axes[0, 0].barh(symbols, scores, color="steelblue")
            axes[0, 0].set_title("Top 10 Stocks by Composite Score")
            axes[0, 0].set_xlabel("Composite Score")

            # Score distribution
            all_scores = [r["composite_score"] for r in results]
            axes[0, 1].hist(all_scores, bins=20, color="lightcoral", alpha=0.7)
            axes[0, 1].set_title("Distribution of Composite Scores")
            axes[0, 1].set_xlabel("Composite Score")
            axes[0, 1].set_ylabel("Number of Stocks")

            # Risk vs Return scatter plot
            volatilities = [r["volatility"] for r in results if r["volatility"] > 0]
            returns = [r["composite_score"] for r in results if r["volatility"] > 0]

            axes[1, 0].scatter(volatilities, returns, alpha=0.6, color="green")
            axes[1, 0].set_title("Risk vs Score Analysis")
            axes[1, 0].set_xlabel("Volatility (%)")
            axes[1, 0].set_ylabel("Composite Score")

            # Score components comparison for top 5
            top_5 = results[:5]
            categories = ["Technical", "Momentum", "Fundamental", "Risk"]

            for i, stock in enumerate(top_5):
                values = [
                    stock["technical_score"],
                    stock["momentum_score"],
                    stock["fundamental_score"],
                    stock["risk_score"],
                ]
                axes[1, 1].plot(categories, values, marker="o", label=stock["symbol"])

            axes[1, 1].set_title("Score Components - Top 5 Stocks")
            axes[1, 1].set_ylabel("Score")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save the chart
            chart_path = os.path.join(self.output_path, f"{base_filename}.png")
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Charts saved to: {chart_path}")

        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")

    @staticmethod
    def send_telegram_report(token: str, chat_id: str, file_path: str):
        """
        Sends a file to a Telegram chat.

        Args:
            token: Telegram bot token
            chat_id: Telegram chat ID
            file_path: Full path to the file to send
        """
        try:
            url = f"https://api.telegram.org/bot{token}/sendDocument"
            with open(file_path, "rb") as file:
                files = {"document": file}
                data = {
                    "chat_id": chat_id,
                    "caption": "📊 *NSE Swing Trading Analysis Results*",
                    "parse_mode": "Markdown",
                }
                response = requests.post(url, data=data, files=files)

            if response.status_code == 200:
                print("✅ File sent successfully to Telegram.")
            else:
                print(f"⚠️ Failed to send file: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠️ Error sending file: {e}")

    def run_analysis(
        self,
        min_score: float = 60,
        save_results: bool = True,
        create_charts: bool = True,
        send_telegram: bool = True,
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        """
        Run the complete analysis with auto-retry capability

        Args:
            min_score: Minimum composite score threshold
            save_results: Whether to save results to files
            create_charts: Whether to create visualization charts
            send_telegram: Whether to send results via Telegram
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                self.logger.info(f"Starting NSE Swing Trading System Analysis (Attempt {retry_count + 1}/{max_retries})...")

                start_time = time.time()

                # Screen stocks
                results = self.screen_stocks(min_score)

                if results:
                    self.logger.info(f"\nANALYSIS COMPLETE!")
                    self.logger.info(f"Found {len(results)} stocks with score >= {min_score}")

                    # Display top 5 results
                    self.logger.info(f"\nTOP 5 RECOMMENDATIONS:")
                    self.logger.info("-" * 80)
                    for i, stock in enumerate(results[:5], 1):
                        self.logger.info(
                            f"{i}. {stock['symbol']:12s} | Score: {stock['composite_score']:6.2f} | "
                            f"Price: ₹{stock['current_price']:8.2f} | R:R: 1:{stock['risk_reward_ratio']:4.2f}"
                        )

                    if save_results:
                        self.save_results(results)

                    if create_charts:
                        self.create_visualization(results)

                    # Send results via Telegram if enabled
                    if send_telegram:
                        TELEGRAM_TOKEN = "7201296239:AAHYnR_Yqy9ixE4KyfJKUxExPQ30e2MrVLc"
                        TELEGRAM_CHAT_ID = "7679750287"
                        
                        # Get the latest CSV file
                        csv_files = [f for f in os.listdir(self.output_path) if f.endswith('.csv') and 'nse_swing_trading_results' in f]
                        if csv_files:
                            latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(self.output_path, x)))
                            csv_path = os.path.join(self.output_path, latest_csv)
                            self.logger.info(f"Sending CSV file to Telegram: {latest_csv}")
                            NSESwingTradingSystem.send_telegram_report(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, csv_path)
                else:
                    self.logger.info(f"No stocks found with score >= {min_score}")

                end_time = time.time()
                self.logger.info(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

                return results, os.path.join(self.output_path, f"nse_swing_trading_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

            except Exception as e:
                last_error = e
                retry_count += 1
                self.logger.error(f"Error during analysis (Attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count < max_retries:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Maximum retry attempts reached. Last error: {str(last_error)}")
                    raise

        return None, None



# Example usage
if __name__ == "__main__":
    try:
        # Initialize the system with current directory as output path
        trading_system = NSESwingTradingSystem(output_path=os.path.dirname(os.path.abspath(__file__)))

        print("\nStarting full run with all stocks...")
        start_time = time.time()

        # Run analysis with minimum score of 60
        results, report_path = trading_system.run_analysis(
            min_score=60,
            save_results=True,
            create_charts=True,
            send_telegram=False,  # Disable Telegram for now
            max_retries=3,
            retry_delay=60
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Print results summary
        if results:
            print(f"\nFull run completed successfully!")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Average time per stock: {execution_time/len(trading_system.smallcap_100_stocks):.2f} seconds")
            print(f"\nTop recommendations:")
            for i, stock in enumerate(results[:5], 1):
                print(f"{i}. {stock['symbol']}: Score {stock['composite_score']:.2f}")
            print(f"\nCheck the output directory for detailed results and charts.")
            print(f"Output directory: {trading_system.output_path}")
        else:
            print("No stocks met the criteria. Try lowering the minimum score threshold.")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Analysis failed. Check the log file for details.")