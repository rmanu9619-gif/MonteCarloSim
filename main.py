import streamlit as st
import numpy as np
import yfinance as yf

# Monte Carlo Simulator class
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
        prob_up = np.mean(results > S0) * 100  # percent chance stock goes up
        avg_return = np.mean(results / S0 - 1) * 100  # average return in %
        return {
            "percent_chance_up": prob_up,
            "average_return_percent": avg_return
        }

# Streamlit UI
st.title("Monte Carlo Stock Simulator 📈")

# User inputs
ticker = st.text_input("Enter stock ticker:", "AAPL").upper()
n_simulations = st.number_input("Number of simulations:", min_value=1000, value=10000, step=1000)

st.write("Choose time horizon:")
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
        # Fetch stock data
        data = yf.download(ticker, period="1y")
        if data.empty:
            st.error("Ticker not found or no data available!")
        else:
            S0 = float(data["Close"].iloc[-1])
            returns = data["Close"].pct_change().dropna()
            mu = float(returns.mean() * 252)
            sigma = float(returns.std() * np.sqrt(252))

            # Stock simulation function
            def stock_sim(T):
                Z = np.random.normal()
                return S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

            # Run Monte Carlo
            sim = MonteCarloSimulator(lambda: stock_sim(T), n_simulations)
            result = sim.summary(S0)

            # Display results
            st.success(f"Simulation complete for {ticker} over {horizon_option}!")
            st.metric("Percent chance it goes up", f"{result['percent_chance_up']:.2f}%")
            st.metric("Average return", f"{result['average_return_percent']:.2f}%")
