import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --------------------- RSI FUNCTIONS ---------------------
def wilder_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rsi = pd.Series(np.zeros(len(prices)), index=prices.index)
    rsi[:period] = 50  # neutral for first values

    for i in range(period, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        rs = avg_gain.iloc[i] / avg_loss.iloc[i] if avg_loss.iloc[i] != 0 else 0
        rsi.iloc[i] = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi(data):
    data['RSI'] = wilder_rsi(data['Close'])
    return data

def color_rsi(val):
    if val < 30:
        return 'background-color: lightgreen'
    elif val < 40:
        return 'background-color: yellow'
    elif val < 50:
        return 'background-color: orange'
    else:
        return ''  # No dark red

# --------------------- MONTE CARLO CLASS ---------------------
class MonteCarloSimulator:
    def __init__(self, experiment_fn, n_simulations=10000, random_seed=None):
        self.experiment_fn = experiment_fn
        self.n_simulations = n_simulations
        if random_seed is not None:
            np.random.seed(random_seed)

    def run(self):
        results = [self.experiment_fn() for _ in range(self.n_simulations)]
        return np.array(results)

    def summary(self, S0):
        results = self.run()
        prob_up = np.mean(results > S0) * 100
        avg_return = np.mean(results / S0 - 1) * 100
        return {
            "percent_chance_up": prob_up,
            "average_return_percent": avg_return
        }

# --------------------- TABS ---------------------
tab1, tab2 = st.tabs(["RSI Dashboard", "Monte Carlo Simulator"])

# --------------------- RSI TAB ---------------------
with tab1:
    st.title("RSI Stock Dashboard 📊")
    ticker_list_rsi = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]  # truncated for example, replace with full list
    ticker_rsi = st.selectbox("Select stock ticker:", ticker_list_rsi)
    if st.button("Calculate RSI"):
        data_rsi = yf.download(ticker_rsi, period="6mo")
        if data_rsi.empty:
            st.error("No data found for this ticker!")
        else:
            rsi_table = calculate_rsi(data_rsi)
            st.dataframe(rsi_table.style.format({"RSI": "{:.2f}"}).applymap(color_rsi, subset=["RSI"]))

# --------------------- MONTE CARLO TAB ---------------------
with tab2:
    st.title("Monte Carlo Stock Simulator 📈")
    ticker_list_mc = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]  # truncated for example, replace with full list
    ticker_mc = st.selectbox("Select stock ticker:", ticker_list_mc)
    n_simulations = st.slider("Number of simulations:", 1000, 50000, 10000, step=1000)
    horizon_option = st.selectbox(
        "Time period:",
        ["1 week", "1 month", "3 months", "6 months", "1 year"]
    )

    horizons = {
        "1 week": 1/52,
        "1 month": 1/12,
        "3 months": 0.25,
        "6 months": 0.5,
        "1 year": 1.0
    }
    T = horizons[horizon_option]

    if st.button("Run Simulation"):
        with st.spinner("Fetching data and running simulations..."):
            data_mc = yf.download(ticker_mc, period="1y")
            if data_mc.empty:
                st.error("Ticker not found or no data available!")
            else:
                S0 = float(data_mc["Close"].iloc[-1])
                returns = data_mc["Close"].pct_change().dropna()
                mu = float(returns.mean() * 252)
                sigma = float(returns.std() * np.sqrt(252))

                def stock_sim(T):
                    Z = np.random.normal()
                    return S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

                sim = MonteCarloSimulator(lambda: stock_sim(T), n_simulations)
                result = sim.summary(S0)

                st.success(f"Simulation complete for {ticker_mc} over {horizon_option}!")
                st.metric("Percent chance it goes up", f"{result['percent_chance_up']:.2f}%")
                st.metric("Average return", f"{result['average_return_percent']:.2f}%")
