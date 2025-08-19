import streamlit as st
import yfinance as yf
import pandas as pd
import time
import numpy as np

# -------------------------
# RSI Code
# -------------------------
tickers = [
    "AAL","AAPL","ACHC","ADBE","AEHR","AEP","AMD","AMGN","AMTX","AMZN","ARCB","AVGO",
    "BECN","BIDU","CAAS","CAKE","CASY","CHNR","CHPT","CMCSA","COST","CPRX","CSCO","CTSH",
    "CZR","DBX","DJCO","DLTR","ETSY","FIZZ","FTNT","GBCI","GEG","GILD","GMAB","GOGO",
    "GOOGL","GRPN","HAS","HBIO","HTLD","ILMN","INTC","IOSP","JBLU","KALU","KDP","LE","LQDA",
    "LULU","LYFT","MANH","MAR","MAT","META","MIDD","MNST","MSEX","MSFT","MTCH","MYGN","NCTY",
    "NTES","NTIC","NVDA","NXPI","ONB","ORLY","OZK","PCAR","PEP","PTON","PYPL","PZZA","QCOM",
    "REGN","RGLD","ROCK","RTC","SBUX","SEDG","SEIC","SFIX","SFM","SIRI","SKYW","SOHU","SWBI",
    "TROW","TSLA","TXN","TXRH","ULTA","URBN","USLM","UTSI","VEON","VRA","VRSK","WBA","WDFC",
    "WEN","YORW","ABBV","ABT","AEO","AFL","ALL","AMC","AMN","ANET","ANF","APAM","APD","APTV",
    "ASGN","ASH","AWK","AXP","AZO","BA","BABA","BAC","BAM","BAX","BBW","BBY","BCS","BEN","BILL",
    "BLK","BMY","BNED","BP","BUD","BURL","BWA","BX","C","CAT","CCJ","CL","CLW","CMG","CNC",
    "CNI","CP","CPB","CRH","CRM","CTVA","CVS","CVX","CYD","D","DAL","DB","DE","DEO","DFS","DG",
    "DIS","DLR","DOC","DOW","DXC","EDR","EDU","EL","EMN","ENB","ET","EXR","F","FCN","FCX","FE",
    "FICO","FL","FMC","FTS","GD","GE","GEO","GIS","GM","GMED","GRMN","GS","GSK","H","HD","HES",
    "HMC","HOG","HRB","HSY","ICE","IMAX","IQV","IRM","JNJ","JPM","K","KEY","KKR","KMI","KMX",
    "KO","KWR","L","LAC","LAZ","LCII","LMT","LOW","LUV","LVS","M","MA","MCD","MCK","MCO","MET",
    "MKC","MOV","MRK","MS","MTB","NCLH","NFG","NGS","NKE","NOC","NOV","NTR","NVO","NVS","OKE",
    "OPY","ORCL","PBH","PCG","PFE","PG","PKX","PLNT","PLOW","PNC","PRU","PSA","PSX","RBA",
    "RCI","RF","RTX","SAP","SAVE","SCHW","SJW","SNA","SNOW","SO","SONY","SPOT","SRE","SUN",
    "SYY","T","TAL","TAP","TCS","TEVA","TGT","THS","TJX","TM","TR","TREX","TRP","TSM","TSN",
    "TU","TWI","TXT","UA","UBER","UBS","UGI","UL","UNFI","UNH","UPS","V","VEEV","VFC","VZ",
    "WFC","WH","WHD","WMT","WNC","WSM","X","XOM","XRX","YUM","ZTO"
]

def wilder_rsi(prices, period=14):
    prices = prices.reset_index(drop=True)  # integer index
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Initialize full-length Series
    avg_gain = pd.Series(index=range(len(prices)), dtype=float)
    avg_loss = pd.Series(index=range(len(prices)), dtype=float)
    
    # First average is simple rolling mean
    avg_gain[period] = gain[:period+1].mean()
    avg_loss[period] = loss[:period+1].mean()
    
    # Wilder smoothing
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.round(2)
    return rsi

@st.cache_data(ttl=86400)
def download_data_in_batches(tickers, batch_size=50):
    all_data = pd.DataFrame()
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="1y")["Close"]
            all_data = pd.concat([all_data, data], axis=1)
        except:
            pass
        time.sleep(1)
    return all_data

def calculate_rsi(data):
    rsi_dict = {}
    for ticker in data.columns:
        prices = data[ticker].dropna()
        if len(prices) >= 14:
            rsi_series = wilder_rsi(prices)
            rsi_dict[ticker] = round(rsi_series.iloc[-1], 2)
        else:
            rsi_dict[ticker] = None
    rsi_table = pd.DataFrame(list(rsi_dict.items()), columns=["Ticker", "RSI"])
    rsi_table["RSI"] = rsi_table["RSI"].round(2)
        
    def rsi_to_color(val):
        if val is None:
            return 'white'
        elif val <= 30:
            return "lightgreen"
        elif val <= 40:
            return "green"
        elif val >= 70:
            return "red"
        else:
            return "white"
        
    rsi_table['Color'] = rsi_table['RSI'].apply(rsi_to_color)
    return rsi_table

# -------------------------
# Monte Carlo Code
# -------------------------
class MonteCarloSimulator:
    def __init__(self, ticker, n_simulations=10000, horizon="1 month", random_seed=None):
            self.ticker = ticker.upper()
            self.n_simulations = n_simulations
            self.horizon = horizon
            if random_seed is not None:
                np.random.seed(random_seed)
    
            # Time horizons in years
            self.horizons = {
                "1 week": 1/52,
                "1 month": 1/12,
                "3 months": 0.25,
                "6 months": 0.5,
                "1 year": 1.0
            }
            self.T = self.horizons.get(horizon, 1/12)  # default 1 month
    
            # Initialize stock data
            self.S0, self.mu, self.sigma = self._fetch_stock_data()
    
    def _fetch_stock_data(self):
        data = yf.download(self.ticker, period="1y")
        if data.empty:
            raise ValueError(f"Ticker {self.ticker} not found or no data available!")
        S0 = float(data["Close"].iloc[-1])
        returns = data["Close"].pct_change().dropna()
        mu = float(returns.mean() * 252)
        sigma = float(returns.std() * np.sqrt(252))
        return S0, mu, sigma

    def _simulate_one(self):
        Z = np.random.normal()
        return self.S0 * np.exp((self.mu - 0.5*self.sigma**2)*self.T + self.sigma*np.sqrt(self.T)*Z)

    def run(self):
        results = [self._simulate_one() for _ in range(self.n_simulations)]
        return np.array(results)

    def summary(self):
        results = self.run()
        prob_up = np.mean(results > self.S0) * 100
        avg_return = np.mean(results / self.S0 - 1) * 100
        return {
            "percent_chance_up": prob_up,
            "average_return_percent": avg_return
        }


# -------------------------
# Streamlit UI with Tabs
# -------------------------
st.title("Stock Dashboard")

tab1, tab2 = st.tabs(["RSI Dashboard", "Monte Carlo Simulation"])

with tab1:
    st.header("RSI Dashboard")
    
    # Download data
    data = download_data_in_batches(tickers)
    
    # Refresh button
    if st.button("Refresh RSI", key="refresh_rsi"):
        st.cache_data.clear()
        data = download_data_in_batches(tickers)
    
    # Calculate RSI
    rsi_table = calculate_rsi(data)
    
    # Search box
    search = st.text_input("Search ticker:", key="search_rsi")
    if search:
        filtered_table = rsi_table[rsi_table['Ticker'].str.contains(search.upper())]
    else:
        filtered_table = rsi_table.copy()
    
    # Color filter dropdown
    color_options = ["All", "lightgreen", "green", "red"]
    selected_color = st.selectbox("Filter by RSI color:", color_options, key="color_rsi")
    if selected_color != "All":
        filtered_table = filtered_table[filtered_table['Color'] == selected_color]
    
    # Display styled table
    def color_rsi(val):
        return f'background-color: {val}'
    
    styled_table = filtered_table.style.applymap(color_rsi, subset=['Color'])
    st.dataframe(styled_table, height=600)

with tab2:
    st.header("Monte Carlo Simulation")
    
    S0 = st.number_input("Initial stock price (S0):", value=100.0)
    mu = st.number_input("Expected return (mu):", value=0.1)
    sigma = st.number_input("Volatility (sigma):", value=0.2)
    steps = st.number_input("Steps (days):", value=252)
    sims = st.number_input("Simulations:", value=1000)
    
    if st.button("Run Monte Carlo", key="run_mc"):
        prices = monte_carlo_simulation(S0, mu, sigma, steps, steps, sims)
        st.line_chart(prices)
