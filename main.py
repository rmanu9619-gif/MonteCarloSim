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
ticker_list = [ "AAL","AAPL","ACHC","ADBE","AEHR","AEP","AMD","AMGN","AMTX","AMZN","ARCB","AVGO",
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
]  # Example list of tickers
ticker = st.selectbox("Select stock ticker:", ticker_list)
n_simulations = st.slider("Number of simulations:", min_value=1000, max_value=50000, value=10000, step=1000)

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
