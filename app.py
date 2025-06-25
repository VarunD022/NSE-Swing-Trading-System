import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from main import NSESwingTradingSystem

# Set page config
st.set_page_config(
    page_title="NSE Swing Trading System",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = NSESwingTradingSystem()

# Title and description
st.title("NSE Swing Trading System")
st.markdown("""
    This application analyzes NSE stocks for swing trading opportunities using technical, fundamental, and sentiment analysis.
    The system evaluates multiple factors to generate a comprehensive trading score for each stock.
""")

# Sidebar controls
st.sidebar.header("Analysis Parameters")
min_score = st.sidebar.slider("Minimum Score", 0, 100, 60, help="Minimum composite score for stock recommendations")
market_condition = st.sidebar.selectbox(
    "Market Condition",
    ["bull", "bear", "neutral"],
    help="Current market condition for adjusting scoring weights"
)

# Function to process stocks in batches
def process_stocks_in_batches():
    batch_size = 20  # Process 20 stocks at a time
    all_stocks = st.session_state.trading_system.smallcap_100_stocks
    total_stocks = len(all_stocks)
    total_batches = (total_stocks + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_stocks)
        batch_stocks = all_stocks[start_idx:end_idx]
        
        status_text.text(f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx + 1}-{end_idx} of {total_stocks} stocks)")
        
        # Process batch
        batch_results = st.session_state.trading_system.analyze_stock_batch(batch_stocks, market_condition)
        st.session_state.results.extend([r for r in batch_results if r and "error" not in r])
        
        # Update progress
        progress = (batch_idx + 1) / total_batches
        progress_bar.progress(progress)
        
        # Show current top results
        if st.session_state.results:
            current_results = sorted(st.session_state.results, key=lambda x: x["composite_score"], reverse=True)
            with results_container.container():
                st.subheader("Current Top Recommendations")
                cols = st.columns(3)
                for i, stock in enumerate(current_results[:3]):
                    with cols[i]:
                        st.metric(
                            label=stock["symbol"],
                            value=f"Score: {stock['composite_score']:.2f}",
                            delta=f"Price: ₹{stock['current_price']:.2f}"
                        )
    
    return st.session_state.results

# Run analysis button
if st.sidebar.button("Run Analysis"):
    st.session_state.results = []
    st.session_state.analysis_started = True
    
    with st.spinner("Starting analysis..."):
        results = process_stocks_in_batches()

# Display results if available
if st.session_state.results:
    st.header("Analysis Results")
    
    # Filter results by minimum score
    filtered_results = [r for r in st.session_state.results if r["composite_score"] >= min_score]
    
    if filtered_results:
        # Convert to DataFrame for easier display
        df = pd.DataFrame(filtered_results)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks Analyzed", len(st.session_state.results))
        with col2:
            st.metric("Stocks Above Threshold", len(filtered_results))
        with col3:
            st.metric("Average Score", f"{df['composite_score'].mean():.2f}")
        
        # Display top recommendations
        st.subheader("Top Recommendations")
        top_stocks = df.nlargest(5, "composite_score")
        for _, stock in top_stocks.iterrows():
            with st.expander(f"{stock['symbol']} - Score: {stock['composite_score']:.2f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Current Price: ₹{stock['current_price']:.2f}")
                    st.write(f"Technical Score: {stock['technical_score']:.2f}")
                    st.write(f"Momentum Score: {stock['momentum_score']:.2f}")
                    st.write(f"Fundamental Score: {stock['fundamental_score']:.2f}")
                with col2:
                    st.write(f"Stop Loss: ₹{stock['stop_loss']:.2f}")
                    st.write(f"Target 1: ₹{stock['target_1']:.2f}")
                    st.write(f"Target 2: ₹{stock['target_2']:.2f}")
                    st.write(f"Risk:Reward: 1:{stock['risk_reward_ratio']:.2f}")
        
        # Display detailed results table
        st.subheader("Detailed Results")
        st.dataframe(
            df[["symbol", "composite_score", "current_price", "technical_score", 
                "momentum_score", "fundamental_score", "risk_reward_ratio"]],
            use_container_width=True
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"swing_trading_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No stocks found with score >= {min_score}. Try lowering the threshold.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Disclaimer: This analysis is for educational purposes only. Please consult with a financial advisor before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True) 
