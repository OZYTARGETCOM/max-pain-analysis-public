import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import csv
import bcrypt
import os
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import socket
from scipy.stats import norm
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import List, Dict, Optional, Tuple

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de APIs ---
FMP_API_KEY = "bQ025fPNVrYcBN4KaExd1N3Xczyk44wM"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
TRADIER_API_KEY = "d0H5QGsma6Bh41VBw6P6lItCBl7D"
TRADIER_BASE_URL = "https://api.tradier.com/v1"

HEADERS_FMP = {"Accept": "application/json"}
HEADERS_TRADIER = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}

# --- Constantes ---
PASSWORDS_FILE = "passwords.csv"
CACHE_TTL = 300
MAX_RETRIES = 5
INITIAL_DELAY = 1
RISK_FREE_RATE = 0.045

# --- Lista de Tickers ---
all_tickers = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "ADBE", "INTC", "NFLX",
    "QCOM", "CSCO", "AMD", "PYPL", "AVGO", "AMAT", "TXN", "MRVL", "INTU", "SHOP",
    "JD", "ZM", "DOCU", "CRWD", "SNOW", "ZS", "PANW", "SPLK", "MDB", "OKTA",
    "ROKU", "ALGN", "ADSK", "DXCM", "TEAM", "PDD", "MELI", "BIDU", "BABA", "NTES",
    "ATVI", "EA", "ILMN", "EXPE", "SIRI", "KLAC", "LRCX", "ASML", "SWKS", "XLNX",
    "WDAY", "TTWO", "VRTX", "REGN", "BIIB", "SGEN", "MAR", "CTSH", "FISV", "MTCH",
    "TTD", "SPLK", "PTON", "DOCS", "UPST", "HIMS", "CRSP", "NVCR", "EXAS", "ARKK",
    "ZS", "TWLO", "U", "HUBS", "VIX", "BILL", "ZI", "GTLB", "NET", "FVRR",
    "TTD", "COIN", "RBLX", "DKNG", "SPOT", "SNAP", "PINS", "MTCH", "LYFT", "GRPN",
    "BRK.B", "JNJ", "V", "PG", "JPM", "HD", "DIS", "MA", "UNH", "PFE", "KO", "PEP",
    "BAC", "WMT", "XOM", "CVX", "ABT", "TMO", "MRK", "MCD", "CAT", "GS", "MMM",
    "RTX", "IBM", "DOW", "GE", "BA", "LMT", "FDX", "T", "VZ", "NKE", "AXP", "ORCL",
    "CSX", "USB", "SPG", "AMT", "PLD", "CCI", "PSA", "CB", "BK", "SCHW", "TFC", "SO",
    "D", "DUK", "NEE", "EXC", "SRE", "AEP", "EIX", "PPL", "PEG", "FE", "AEE", "AES",
    "ETR", "XEL", "AWK", "WEC", "ED", "ES", "CNP", "CMS", "DTE", "EQT", "OGE",
    "OKE", "SWX", "WMB", "APA", "DVN", "FANG", "MRO", "PXD", "HAL", "SLB", "COP",
    "CVX", "XOM", "PSX", "MPC", "VLO", "HES", "OXY", "EOG", "KMI", "WES", "DJT", "BITX", "SMCI", "ENPH",
    "PLTR", "ROKU", "SQ", "AFRM", "UPST", "FVRR", "ETSY", "NET", "DDOG", "TWLO",
    "U", "HUBS", "DOCN", "GTLB", "SMAR", "PATH", "COUP", "ASAN", "RBLX", "DKNG",
    "BILL", "ZI", "TTD", "CRSP", "NVCR", "EXAS", "ARKK", "MTCH", "LYFT", "GRPN",
    "BB", "CLF", "FUBO", "CLOV", "NNDM", "SKLZ", "SPCE", "SNDL", "WKHS", "GME",
    "AMC", "BBBY", "APRN", "SPWR", "RUN", "FCEL", "PLUG", "SOLO", "KNDI", "XPEV",
    "LI", "NIO", "RIDE", "NKLA", "QS", "LCID", "FSR", "PSNY", "GOEV", "WKHS",
    "VRM", "BABA", "JD", "PDD", "BIDU", "TCEHY", "NTES", "IQ", "HUYA", "DOYU",
    "EDU", "TAL", "ZH", "DIDI", "YMM", "BILI", "PDD", "LU", "QD", "FINV",
    "OCGN", "NVTA", "CRSP", "BEAM", "EDIT", "NTLA", "PACB", "TWST", "FLGT", "FATE"
]

# --- Autenticación ---
def initialize_passwords_file():
    if not os.path.exists(PASSWORDS_FILE):
        with open(PASSWORDS_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            passwords = [
                ["abc123", 0, ""], ["def456", 0, ""], ["ghi789", 0, ""], ["jkl010", 0, ""],
                ["mno345", 0, ""], ["pqr678", 0, ""], ["stu901", 0, ""], ["vwx234", 0, ""],
                ["yz1234", 0, ""], ["abcd56", 0, ""], ["efgh78", 0, ""], ["ijkl90", 0, ""],
                ["mnop12", 0, ""], ["qrst34", 0, ""], ["uvwx56", 0, ""], ["yzab78", 0, ""],
                ["cdef90", 0, ""], ["ghij12", 0, ""], ["news34", 0, ""], ["opqr56", 0, ""],
            ]
            writer.writerows(passwords)

def get_local_ip():
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return None

def load_passwords():
    passwords = {}
    try:
        with open(PASSWORDS_FILE, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 3:
                    password, status, ip = row
                    passwords[password] = {"status": int(status), "ip": ip}
    except Exception:
        pass
    return passwords

def save_passwords(passwords):
    with open(PASSWORDS_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        for password, data in passwords.items():
            writer.writerow([password, data["status"], data["ip"]])

def authenticate_password(input_password):
    local_ip = get_local_ip()
    if not local_ip:
        st.error("No se pudo obtener la IP local.")
        return False
    passwords = load_passwords()
    if input_password in passwords:
        password_data = passwords[input_password]
        if password_data["status"] == 0:
            passwords[input_password]["status"] = 1
            passwords[input_password]["ip"] = local_ip
            save_passwords(passwords)
            return True
        elif password_data["status"] == 1 and password_data["ip"] == local_ip:
            return True
        elif password_data["status"] == 1 and password_data["ip"] != local_ip:
            st.warning("⚠️ Esta contraseña ya ha sido usada desde otra IP.")
            return False
    return False

initialize_passwords_file()
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Acceso VIP")
    password = st.text_input("Ingresa tu contraseña", type="password")
    if st.button("Iniciar Sesión"):
        if authenticate_password(password):
            st.session_state["authenticated"] = True
            st.success("✅ Acceso concedido!")
    else:
        st.error("❌ Acceso solo para clientes VIP.")
    st.stop()

# --- Funciones de API Optimizadas ---
def fetch_api_data(url: str, params: Dict, headers: Dict, source: str) -> Optional[Dict]:
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=5)  # Timeout reducido
            response.raise_for_status()
            logger.info(f"{source} fetch success: {response.text[:100]}...")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"{source} error attempt {attempt + 1}: {e}")
            if attempt == MAX_RETRIES - 1:
                return None
            delay = INITIAL_DELAY * (2 ** attempt) + np.random.uniform(0, 0.1)  # Reintentos más rápidos
            time.sleep(delay)
    return None

@st.cache_data(ttl=CACHE_TTL)
def get_current_price(ticker: str) -> float:
    url = f"{FMP_BASE_URL}/quote/{ticker}"
    params = {"apikey": FMP_API_KEY}
    data = fetch_api_data(url, params, HEADERS_FMP, "FMP")
    try:
        if data and isinstance(data, list) and len(data) > 0:
            price = float(data[0].get("price", 0.0))
            if price > 0:
                logger.info(f"FMP price for {ticker}: ${price:.2f}")
                return price
    except (ValueError, TypeError) as e:
        logger.error(f"FMP price error: {e}")
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    params = {"symbols": ticker}
    data = fetch_api_data(url, params, HEADERS_TRADIER, "Tradier")
    try:
        if data and 'quotes' in data and 'quote' in data['quotes']:
            price = float(data['quotes']['quote'].get("last", 0.0))
            logger.info(f"Tradier price for {ticker}: ${price:.2f}")
            return price
        return 0.0
    except (ValueError, TypeError) as e:
        logger.error(f"Tradier price error: {e}")
        return 0.0

@st.cache_data(ttl=CACHE_TTL)
def get_expiration_dates(ticker: str) -> List[str]:
    if not ticker or not ticker.isalnum():
        return []
    url = f"{TRADIER_BASE_URL}/markets/options/expirations"
    params = {"symbol": ticker}
    data = fetch_api_data(url, params, HEADERS_TRADIER, "Tradier")
    if (data is not None and 
        isinstance(data, dict) and 
        'expirations' in data and 
        data['expirations'] is not None and 
        isinstance(data['expirations'], dict) and 
        'date' in data['expirations'] and 
        data['expirations']['date'] is not None):
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            expiration_dates = [date_str for date_str in sorted(data['expirations']['date'])
                                if (datetime.strptime(date_str, "%Y-%m-%d") - today).days >= 0]
            logger.info(f"Found {len(expiration_dates)} expiration dates for {ticker}")
            return expiration_dates
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing expiration dates for {ticker}: {str(e)}")
            return []
    logger.error(f"No valid expiration dates found for {ticker}")
    return []

@st.cache_data(ttl=CACHE_TTL)
def get_options_data(ticker: str, expiration_date: str) -> List[Dict]:
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {"symbol": ticker, "expiration": expiration_date, "greeks": "true"}
    data = fetch_api_data(url, params, HEADERS_TRADIER, "Tradier")
    if data and 'options' in data and 'option' in data['options']:
        return data['options']['option']
    return []

@st.cache_data(ttl=CACHE_TTL)
def get_historical_prices_combined(symbol, period="daily", limit=30):
    """Obtener precios históricos combinando FMP y Tradier para máxima velocidad y fiabilidad."""
    url_fmp = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
    params_fmp = {"apikey": FMP_API_KEY, "timeseries": limit}
    data_fmp = fetch_api_data(url_fmp, params_fmp, HEADERS_FMP, "FMP")
    if data_fmp and "historical" in data_fmp:
        prices = [float(day["close"]) for day in data_fmp["historical"]]
        volumes = [int(day["volume"]) for day in data_fmp["historical"]]
        if prices and volumes:
            return prices, volumes

    url_tradier = f"{TRADIER_BASE_URL}/markets/history"
    params_tradier = {"symbol": symbol, "interval": period, "start": (datetime.now().date() - timedelta(days=limit)).strftime("%Y-%m-%d")}
    data_tradier = fetch_api_data(url_tradier, params_tradier, HEADERS_TRADIER, "Tradier")
    if data_tradier and "history" in data_tradier and "day" in data_tradier["history"]:
        prices = [float(day["close"]) for day in data_tradier["history"]["day"]]
        volumes = [int(day["volume"]) for day in data_tradier["history"]["day"]]
        return prices, volumes
    
    logger.warning(f"No historical data for {symbol} from either API")
    return [], []

@st.cache_data(ttl=CACHE_TTL)
def get_stock_list_combined():
    """Obtener lista de acciones combinando FMP y Tradier."""
    try:
        response = requests.get(f"{FMP_BASE_URL}/stock-screener", params={"apikey": FMP_API_KEY, "marketCapMoreThan": 1_000_000_000, "volumeMoreThan": 500_000, "priceMoreThan": 10, "priceLessThan": 100, "betaMoreThan": 1})
        response.raise_for_status()
        data = response.json()
        stock_list = [stock["symbol"] for stock in data]
        if stock_list:
            return stock_list[:200]  # Limitar para rapidez inicial
    except Exception as e:
        logger.error(f"FMP stock list failed: {str(e)}")
    
    tickers_batch = all_tickers[:200]
    url_tradier = f"{TRADIER_BASE_URL}/markets/quotes"
    params_tradier = {"symbols": ",".join(tickers_batch)}
    data_tradier = fetch_api_data(url_tradier, params_tradier, HEADERS_TRADIER, "Tradier")
    if data_tradier and "quotes" in data_tradier and "quote" in data_tradier["quotes"]:
        quotes = data_tradier["quotes"]["quote"]
        if isinstance(quotes, dict):
            quotes = [quotes]
        return [quote["symbol"] for quote in quotes if quote.get("last", 0) > 10 and quote.get("volume", 0) > 500_000]
    
    return all_tickers[:100]

# --- Funciones de Análisis ---
def analyze_contracts(ticker, expiration, current_price):
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {"symbol": ticker, "expiration": expiration, "greeks": True}
    response = requests.get(url, headers=HEADERS_TRADIER, params=params)
    if response.status_code != 200:
        st.error("Error retrieving option contracts.")
        return pd.DataFrame()
    options = response.json().get("options", {}).get("option", [])
    if not options:
        st.warning("No contracts available.")
        return pd.DataFrame()
    df = pd.DataFrame(options)
    for col in ['strike', 'option_type', 'open_interest', 'volume', 'bid', 'ask', 'last_volume', 'trade_date', 'bid_exchange', 'delta', 'gamma', 'break_even']:
        if col not in df.columns:
            df[col] = 0
    df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
    df['break_even'] = df.apply(lambda row: row['strike'] + row['bid'] if row['option_type'] == 'call' else row['strike'] - row['bid'], axis=1)
    return df

def style_and_sort_table(df):
    ordered_columns = ['strike', 'option_type', 'open_interest', 'volume', 'trade_date', 'bid', 'ask', 'last_volume', 'bid_exchange', 'delta', 'gamma', 'break_even']
    df = df.sort_values(by=['volume', 'open_interest'], ascending=[False, False]).head(10)
    df = df[ordered_columns]
    def highlight_row(row):
        color = 'background-color: green; color: white;' if row['option_type'] == 'call' else 'background-color: red; color: white;'
        return [color] * len(row)
    return df.style.apply(highlight_row, axis=1).format({
        'strike': '{:.2f}', 'bid': '${:.2f}', 'ask': '${:.2f}', 'last_volume': '{:,}', 'open_interest': '{:,}', 'delta': '{:.2f}', 'gamma': '{:.2f}', 'break_even': '${:.2f}'
    })

def select_best_contracts(df, current_price):
    if df.empty:
        return None, None
    df['strike_diff'] = abs(df['strike'] - current_price)
    closest_contract = df.sort_values(by=['strike_diff', 'volume', 'open_interest'], ascending=[True, False, False]).iloc[0]
    otm_calls = df[(df['option_type'] == 'call') & (df['strike'] > current_price) & (df['ask'] < 5)]
    otm_puts = df[(df['option_type'] == 'put') & (df['strike'] < current_price) & (df['ask'] < 5)]
    if not otm_calls.empty or not otm_puts.empty:
        economic_df = pd.concat([otm_calls, otm_puts])
        economic_contract = economic_df.sort_values(by=['volume', 'open_interest'], ascending=[False, False]).iloc[0]
    else:
        economic_contract = None
    return closest_contract, economic_contract

def calculate_max_pain(df):
    if df.empty:
        return None, pd.DataFrame()
    strikes = df['strike'].unique()
    max_pain_data = []
    for strike in strikes:
        call_losses = ((strike - df[df['option_type'] == 'call']['strike']).clip(lower=0) * 
                       df[df['option_type'] == 'call']['open_interest']).sum()
        put_losses = ((df[df['option_type'] == 'put']['strike'] - strike).clip(lower=0) * 
                      df[df['option_type'] == 'put']['open_interest']).sum()
        total_loss = call_losses + put_losses
        max_pain_data.append({'strike': strike, 'total_loss': total_loss})
    max_pain_df = pd.DataFrame(max_pain_data)
    if max_pain_df.empty:
        return None, max_pain_df
    max_pain_strike = max_pain_df.loc[max_pain_df['total_loss'].idxmin()]
    return max_pain_strike, max_pain_df.sort_values(by='total_loss', ascending=True)

def calculate_support_resistance_mid(max_pain_table, current_price):
    puts = max_pain_table[max_pain_table['strike'] <= current_price]
    calls = max_pain_table[max_pain_table['strike'] > current_price]
    support_level = puts.loc[puts['total_loss'].idxmin()]['strike'] if not puts.empty else current_price
    resistance_level = calls.loc[calls['total_loss'].idxmin()]['strike'] if not calls.empty else current_price
    mid_level = (support_level + resistance_level) / 2
    return support_level, resistance_level, mid_level

def plot_max_pain_histogram_with_levels(max_pain_table, current_price):
    support_level, resistance_level, mid_level = calculate_support_resistance_mid(max_pain_table, current_price)
    max_pain_table['loss_category'] = max_pain_table['total_loss'].apply(
        lambda x: 'High Loss' if x > max_pain_table['total_loss'].quantile(0.75) else ('Low Loss' if x < max_pain_table['total_loss'].quantile(0.25) else 'Neutral')
    )
    color_map = {'High Loss': '#FF5733', 'Low Loss': '#28A745', 'Neutral': 'rgba(128,128,128,0.3)'}
    fig = px.bar(max_pain_table, x='strike', y='total_loss', title="Max Pain Histogram with Levels",
                 labels={'total_loss': 'Total Loss', 'strike': 'Strike Price'}, color='loss_category', color_discrete_map=color_map)
    fig.update_layout(xaxis_title="Strike Price", yaxis_title="Total Loss", template="plotly_white", font=dict(size=14, family="Open Sans"),
                      title=dict(text="📊 Max Pain Analysis", font=dict(size=18), x=0.5), hovermode="x",
                      yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="#FFFF00", spikethickness=1.5))
    mean_loss = max_pain_table['total_loss'].mean()
    fig.add_hline(y=mean_loss, line_width=1, line_dash="dash", line_color="#00FF00", annotation_text=f"Mean Loss: {mean_loss:.2f}", annotation_position="top right", annotation_font=dict(color="#00FF00", size=12))
    fig.add_vline(x=support_level, line_width=1, line_dash="dot", line_color="#1E90FF", annotation_text=f"Support: {support_level:.2f}", annotation_position="top left", annotation_font=dict(color="#1E90FF", size=10))
    fig.add_vline(x=resistance_level, line_width=1, line_dash="dot", line_color="#FF4500", annotation_text=f"Resistance: {resistance_level:.2f}", annotation_position="top right", annotation_font=dict(color="#FF4500", size=10))
    fig.add_vline(x=mid_level, line_width=1, line_dash="solid", line_color="#FFD700", annotation_text=f"Mid Level: {mid_level:.2f}", annotation_position="top right", annotation_font=dict(color="#FFD700", size=8))
    return fig

def get_option_chains(ticker, expiration):
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {"symbol": ticker, "expiration": expiration, "greeks": True}
    response = requests.get(url, headers=HEADERS_TRADIER, params=params)
    if response.status_code == 200:
        return response.json().get("options", {}).get("option", [])
    st.error("Error retrieving option chains.")
    return []

def calculate_score(df, current_price, volatility=0.2):
    df['score'] = (df['open_interest'] * df['volume']) / (abs(df['strike'] - current_price) + volatility)
    return df.sort_values(by='score', ascending=False)

def display_cards(df):
    top_5_vol = df.sort_values(by='volume', ascending=False).head(5)
    st.markdown("### Top 5")
    for i, row in top_5_vol.iterrows():
        st.markdown(f"""
        **Strike:** {row['strike']}  
        **Type:** {'Call' if row['option_type'] == 'call' else 'Put'}  
        **Volume:** {row['volume']}  
        **Open Interest:** {row['open_interest']}  
        **Score:** {row['score']:.2f}  
        """)

def plot_histogram(df):
    fig = px.bar(df, x='strike', y='score', color='option_type', title="Score by Strike (Calls and Puts)",
                 labels={'score': 'Relevance Score', 'strike': 'Strike Price'}, text='score',
                 color_discrete_map={'call': '#00FF00', 'put': '#FF00FF'})
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker=dict(line=dict(width=0.5, color='black')))
    fig.update_layout(plot_bgcolor='black', font=dict(color='white', size=12), xaxis=dict(showgrid=True, gridcolor='gray'),
                      yaxis=dict(showgrid=True, gridcolor='gray'), xaxis_title="Strike Price", yaxis_title="Relevance Score")
    support_level = df['strike'].iloc[0]
    resistance_level = df['strike'].iloc[-1]
    fig.add_hline(y=support_level, line_width=1, line_dash="dot", line_color="#1E90FF", annotation_text=f"Support: {support_level:.2f}", annotation_position="bottom left", annotation_font=dict(size=10, color="#1E90FF"))
    fig.add_hline(y=resistance_level, line_width=1, line_dash="dot", line_color="#FF4500", annotation_text=f"Resistance: {resistance_level:.2f}", annotation_position="top left", annotation_font=dict(size=10, color="#FF4500"))
    return fig

def detect_touched_strikes(strikes, historical_prices):
    touched_strikes = set()
    cleaned_prices = [float(p) for p in historical_prices if isinstance(p, (int, float, str)) and str(p).replace('.', '', 1).isdigit()]
    if len(cleaned_prices) < 2:
        return touched_strikes
    for strike in strikes:
        for i in range(1, len(cleaned_prices)):
            if (cleaned_prices[i-1] < strike <= cleaned_prices[i]) or (cleaned_prices[i-1] > strike >= cleaned_prices[i]):
                touched_strikes.add(strike)
    return touched_strikes

def calculate_max_pain_optimized(options_data):
    if not options_data:
        return None
    strikes = {}
    for option in options_data:
        strike = float(option["strike"])
        oi = int(option.get("open_interest", 0) or 0)
        volume = int(option.get("volume", 0) or 0)
        option_type = option["option_type"].upper()
        if strike not in strikes:
            strikes[strike] = {"CALL": {"OI": 0, "Volume": 0}, "PUT": {"OI": 0, "Volume": 0}}
        strikes[strike][option_type]["OI"] += oi
        strikes[strike][option_type]["Volume"] += volume
    strike_prices = sorted(strikes.keys())
    total_losses = {}
    for strike in strike_prices:
        loss_call = sum((strikes[s]["CALL"]["OI"] + strikes[s]["CALL"]["Volume"]) * max(0, s - strike) for s in strike_prices)
        loss_put = sum((strikes[s]["PUT"]["OI"] + strikes[s]["PUT"]["Volume"]) * max(0, strike - s) for s in strike_prices)
        total_losses[strike] = loss_call + loss_put
    return min(total_losses, key=total_losses.get) if total_losses else None

def gamma_exposure_chart(processed_data, current_price, touched_strikes):
    strikes = sorted(processed_data.keys())
    gamma_calls = [processed_data[s]["CALL"]["OI"] * processed_data[s]["CALL"]["Gamma"] * current_price for s in strikes]
    gamma_puts = [-processed_data[s]["PUT"]["OI"] * processed_data[s]["PUT"]["Gamma"] * current_price for s in strikes]
    call_colors = ["grey" if s in touched_strikes else "#7DF9FF" for s in strikes]
    put_colors = ["orange" if s in touched_strikes else "red" for s in strikes]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=strikes, y=gamma_calls, name="Gamma CALL", marker=dict(color=call_colors),
                         hovertemplate="<b>Strike:</b> %{x}<br><b>Gamma CALL:</b> %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(x=strikes, y=gamma_puts, name="Gamma PUT", marker=dict(color=put_colors),
                         hovertemplate="<b>Strike:</b> %{x}<br><b>Gamma PUT:</b> %{y:.2f}<extra></extra>"))
    
    # Añadir línea vertical para Current Price (delgada, con label transparente)
    y_min = min(gamma_calls + gamma_puts) * 1.1
    y_max = max(gamma_calls + gamma_puts) * 1.1
    fig.add_trace(go.Scatter(x=[current_price, current_price], y=[y_min, y_max], mode="lines",
                             line=dict(width=1, dash="dot", color="#39FF14"),
                             name="Current Price",
                             hovertemplate=f"Current Price: ${current_price:.2f}<br>",
                             showlegend=False))

    # Configuración del layout (sin líneas grises)
    fig.update_traces(hoverlabel=dict(bgcolor="rgba(30,30,30,0.9)", bordercolor="white", font=dict(color="white", size=12)))
    fig.update_layout(title="|GAMMA EXPOSURE|", xaxis_title="Strike", yaxis_title="Gamma Exposure", 
                      template="plotly_dark", hovermode="x unified")
    return fig

def plot_skew_analysis_with_totals(options_data, current_price=None):
    strikes = [float(option["strike"]) for option in options_data]
    iv = [float(option.get("implied_volatility", 0)) * 100 for option in options_data]
    option_type = [option["option_type"].upper() for option in options_data]
    open_interest = [int(option.get("open_interest", 0)) for option in options_data]
    total_calls = sum(oi for oi, ot in zip(open_interest, option_type) if ot == "CALL")
    total_puts = sum(oi for oi, ot in zip(open_interest, option_type) if ot == "PUT")
    total_volume_calls = sum(int(option.get("volume", 0)) for option in options_data if option["option_type"].upper() == "CALL")
    total_volume_puts = sum(int(option.get("volume", 0)) for option in options_data if option["option_type"].upper() == "PUT")
    adjusted_iv = [iv[i] + (open_interest[i] * 0.01) if option_type[i] == "CALL" else -(iv[i] + (open_interest[i] * 0.01)) for i in range(len(iv))]
    skew_df = pd.DataFrame({"Strike": strikes, "Adjusted IV (%)": adjusted_iv, "Option Type": option_type, "Open Interest": open_interest})
    fig = px.scatter(skew_df, x="Strike", y="Adjusted IV (%)", color="Option Type", size="Open Interest",
                     custom_data=["Strike", "Option Type", "Open Interest", "Adjusted IV (%)"],
                     title=f"IV Analysis<br><span style='font-size:16px;'> CALLS: {total_calls} | PUTS: {total_puts} | VC {total_volume_calls} | VP {total_volume_puts}</span>",
                     labels={"Option Type": "Contract Type"}, color_discrete_map={"CALL": "blue", "PUT": "red"})
    fig.update_traces(hovertemplate="<b>Strike:</b> %{customdata[0]:.2f}<br><b>Type:</b> %{customdata[1]}<br><b>Open Interest:</b> %{customdata[2]:,}<br><b>Adjusted IV:</b> %{customdata[3]:.2f}%")
    fig.update_layout(xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)", legend_title="Option Type", template="plotly_white", title_x=0.5)

    if current_price is not None and options_data:
        strikes_dict = {}
        for option in options_data:
            strike = float(option["strike"])
            oi = int(option.get("open_interest", 0) or 0)
            opt_type = option["option_type"].upper()
            if strike not in strikes_dict:
                strikes_dict[strike] = {"CALL": 0, "PUT": 0}
            strikes_dict[strike][opt_type] += oi
        strike_prices = sorted(strikes_dict.keys())
        total_losses = {}
        for strike in strike_prices:
            loss_call = sum((strikes_dict[s]["CALL"] * max(0, s - strike)) for s in strike_prices)
            loss_put = sum((strikes_dict[s]["PUT"] * max(0, strike - s)) for s in strike_prices)
            total_losses[strike] = loss_call + loss_put
        max_pain = min(total_losses, key=total_losses.get) if total_losses else None

        avg_iv_calls = sum(iv[i] + (open_interest[i] * 0.01) for i, ot in enumerate(option_type) if ot == "CALL") / max(1, sum(1 for ot in option_type if ot == "CALL"))
        avg_iv_puts = sum(-(iv[i] + (open_interest[i] * 0.01)) for i, ot in enumerate(option_type) if ot == "PUT") / max(1, sum(1 for ot in option_type if ot == "PUT"))

        call_open_interest = total_calls
        put_open_interest = total_puts
        scale_factor = 5000
        call_size = max(5, min(30, call_open_interest / scale_factor))
        put_size = max(5, min(30, put_open_interest / scale_factor))

        if current_price is not None and max_pain is not None:
            calls_data = [opt for opt in options_data if opt["option_type"].upper() == "CALL"]
            puts_data = [opt for opt in options_data if opt["option_type"].upper() == "PUT"]
            closest_call = min(calls_data, key=lambda x: abs(float(x["strike"]) - current_price), default=None)
            closest_put = min(puts_data, key=lambda x: abs(float(x["strike"]) - current_price), default=None)

            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            exp_date = datetime.strptime(options_data[0].get("expiration_date") or options_data[0].get("expirationDate"), "%Y-%m-%d")
            days_to_expiration = (exp_date - today).days

            if closest_call:
                call_strike = float(closest_call["strike"])
                call_data = {
                    call_strike: {
                        'bid': float(closest_call.get("bid", 0)),
                        'ask': float(closest_call.get("ask", 0)),
                        'delta': float(closest_call.get("greeks", {}).get("delta", 0.5)),
                        'gamma': float(closest_call.get("greeks", {}).get("gamma", 0.02)),
                        'theta': float(closest_call.get("greeks", {}).get("theta", -0.01)),
                        'iv': float(closest_call.get("implied_volatility", 0.2)),
                        'open_interest': int(closest_call.get("open_interest", 0)),
                        'intrinsic': max(current_price - call_strike, 0)
                    }
                }
                rr_calls, profit_calls, prob_otm_calls, _ = calculate_special_monetization(call_data, current_price, days_to_expiration)
                percent_change_calls = ((current_price - max_pain) / max_pain) * 100 if max_pain != 0 else 0
                call_loss = abs(current_price - max_pain) * total_calls if current_price < max_pain else (current_price - max_pain) * total_calls
                potential_move_calls = abs(current_price - max_pain)
                direction_calls = "Down" if current_price > max_pain else "Up"
            else:
                rr_calls, profit_calls, prob_otm_calls, percent_change_calls, call_loss, potential_move_calls, direction_calls = 0, 0, 0, 0, 0, 0, "N/A"

            if closest_put:
                put_strike = float(closest_put["strike"])
                put_data = {
                    put_strike: {
                        'bid': float(closest_put.get("bid", 0)),
                        'ask': float(closest_put.get("ask", 0)),
                        'delta': float(closest_put.get("greeks", {}).get("delta", -0.5)),
                        'gamma': float(closest_put.get("greeks", {}).get("gamma", 0.02)),
                        'theta': float(closest_put.get("greeks", {}).get("theta", -0.01)),
                        'iv': float(closest_put.get("implied_volatility", 0.2)),
                        'open_interest': int(closest_put.get("open_interest", 0)),
                        'intrinsic': max(put_strike - current_price, 0)
                    }
                }
                rr_puts, profit_puts, prob_otm_puts, _ = calculate_special_monetization(put_data, current_price, days_to_expiration)
                percent_change_puts = ((max_pain - current_price) / max_pain) * 100 if max_pain != 0 else 0
                put_loss = abs(max_pain - current_price) * total_puts if current_price > max_pain else (max_pain - current_price) * total_puts
                potential_move_puts = abs(max_pain - current_price)
                direction_puts = "Up" if current_price < max_pain else "Down"
            else:
                rr_puts, profit_puts, prob_otm_puts, percent_change_puts, put_loss, potential_move_puts, direction_puts = 0, 0, 0, 0, 0, 0, "N/A"

            if call_open_interest > 0 and closest_call:
                fig.add_scatter(x=[current_price], y=[avg_iv_calls], mode="markers", name="Current Price (CALLs)",
                                marker=dict(size=call_size, color="yellow", opacity=0.45, symbol="circle"),
                                hovertemplate=(f"Current Price (CALLs): {current_price:.2f}<br>"
                                               f"Adjusted IV: {avg_iv_calls:.2f}%<br>"
                                               f"Open Interest: {call_open_interest:,}<br>"
                                               f"% to Max Pain: {percent_change_calls:.2f}%<br>"
                                               f"R/R: {rr_calls:.2f}<br>"
                                               f"Est. Loss: ${call_loss:,.2f}<br>"
                                               f"Potential Move: ${potential_move_calls:.2f}<br>"
                                               f"Direction: {direction_calls}"))

            if put_open_interest > 0 and closest_put:
                fig.add_scatter(x=[current_price], y=[avg_iv_puts], mode="markers", name="Current Price (PUTs)",
                                marker=dict(size=put_size, color="yellow", opacity=0.45, symbol="circle"),
                                hovertemplate=(f"Current Price (PUTs): {current_price:.2f}<br>"
                                               f"Adjusted IV: {avg_iv_puts:.2f}%<br>"
                                               f"Open Interest: {put_open_interest:,}<br>"
                                               f"% to Max Pain: {percent_change_puts:.2f}%<br>"
                                               f"R/R: {rr_puts:.2f}<br>"
                                               f"Est. Loss: ${put_loss:,.2f}<br>"
                                               f"Potential Move: ${potential_move_puts:.2f}<br>"
                                               f"Direction: {direction_puts}"))

        if max_pain is not None:
            fig.add_scatter(x=[max_pain], y=[0], mode="markers", name="Max Pain",
                            marker=dict(size=15, color="white", symbol="circle"),
                            hovertemplate=f"Max Pain: {max_pain:.2f}")

    return fig, total_calls, total_puts

def fetch_google_news(keywords):
    base_url = "https://www.google.com/search"
    query = "+".join(keywords)
    params = {"q": query, "tbm": "nws", "tbs": "qdr:h"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        news = []
        articles = soup.select("div.dbsr") or soup.select("div.Gx5Zad.fP1Qef.xpd.EtOod.pkphOe")
        for article in articles[:20]:
            title_tag = article.select_one("div.JheGif.nDgy9d") or article.select_one("div.BNeawe.vvjwJb.AP7Wnd")
            link_tag = article.a
            if title_tag and link_tag:
                title = title_tag.text.strip()
                link = link_tag["href"]
                time_tag = article.select_one("span.WG9SHc")
                time_posted = time_tag.text if time_tag else "Just now"
                news.append({"title": title, "link": link, "time": time_posted})
        return news
    except Exception as e:
        st.warning(f"Error fetching Google News: {e}")
        return []

def fetch_bing_news(keywords):
    base_url = "https://www.bing.com/news/search"
    query = " ".join(keywords)
    params = {"q": query, "qft": "+filterui:age-lt24h"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        news = []
        articles = soup.select("a.title")
        for article in articles[:20]:
            title = article.text.strip()
            link = article["href"]
            news.append({"title": title, "link": link, "time": "Recently"})
        return news
    except Exception as e:
        st.warning(f"Error fetching Bing News: {e}")
        return []

def fetch_instagram_posts(keywords):
    base_url = "https://www.instagram.com/explore/tags/"
    posts = []
    for keyword in keywords:
        if keyword.startswith("#"):
            try:
                url = f"{base_url}{keyword[1:]}/"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code != 200:
                    continue
                soup = BeautifulSoup(response.text, "html.parser")
                articles = soup.select("div.v1Nh3.kIKUG._bz0w a")
                for article in articles[:20]:
                    link = "https://www.instagram.com" + article["href"]
                    posts.append({"title": "Instagram Post", "link": link, "time": "Recently"})
            except Exception as e:
                st.warning(f"Error fetching Instagram posts for {keyword}: {e}")
    return posts

def fetch_batch_stock_data(tickers):
    tickers_str = ",".join(tickers)
    url = f"{TRADIER_BASE_URL}/markets/quotes"
    params = {"symbols": tickers_str}
    response = requests.get(url, headers=HEADERS_TRADIER, params=params)
    if response.status_code == 200:
        data = response.json().get("quotes", {}).get("quote", [])
        if isinstance(data, dict):
            data = [data]
        return [{"Ticker": item.get("symbol", ""), "Price": item.get("last", 0), "Change (%)": item.get("change_percentage", 0),
                 "Volume": item.get("volume", 0), "Average Volume": item.get("average_volume", 1),
                 "IV": item.get("implied_volatility", None), "HV": item.get("historical_volatility", None),
                 "Previous Close": item.get("prev_close", 0)} for item in data]
    st.error(f"Error fetching batch data: {response.status_code}")
    return []

def calculate_explosive_movers(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df["IV"] = pd.to_numeric(df["IV"], errors='coerce').fillna(0)
    df["HV"] = pd.to_numeric(df["HV"], errors='coerce').fillna(0)
    df["Average Volume"] = pd.to_numeric(df["Average Volume"], errors='coerce').replace(0, np.nan)
    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Explosión"] = df["Volumen Relativo"] * df["Change (%)"].abs()
    df["Score"] = df["Explosión"] + (df["IV"] * 0.5)
    return df.sort_values("Score", ascending=False).head(3)

def calculate_options_activity(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df["IV"] = pd.to_numeric(df["IV"], errors='coerce').fillna(0)
    df["Average Volume"] = pd.to_numeric(df["Average Volume"], errors='coerce').replace(0, np.nan)
    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Options Activity"] = df["Volumen Relativo"] * df["IV"]
    return df.sort_values("Options Activity", ascending=False).head(3)

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(prices, period=20):
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def scan_stock(symbol, scan_type, breakout_period=10, volume_threshold=2.0):
    prices, volumes = get_historical_prices_combined(symbol)
    if len(prices) <= breakout_period or len(volumes) == 0:
        return None
    rsi = calculate_rsi(prices)
    sma = calculate_sma(prices)
    avg_volume = np.mean(volumes)
    current_volume = volumes[-1]
    recent_high = max(prices[-breakout_period:])
    recent_low = min(prices[-breakout_period:])
    last_price = prices[-1]
    near_support = abs(last_price - recent_low) / recent_low <= 0.05
    near_resistance = abs(last_price - recent_high) / recent_high <= 0.05
    breakout_type = "Up" if last_price > recent_high else "Down" if last_price < recent_low else None
    possible_change = (recent_low - last_price) / last_price * 100 if near_support else (recent_high - last_price) / last_price * 100 if near_resistance else None
    
    if scan_type == "Bullish (Upward Momentum)" and sma is not None and last_price > sma and rsi is not None and rsi < 70:
        return {"Symbol": symbol, "Last Price": last_price, "SMA": round(sma, 2), "RSI": round(rsi, 2), "Near Support": near_support, "Near Resistance": near_resistance, "Volume": current_volume, "Breakout Type": breakout_type, "Possible Change (%)": round(possible_change, 2) if possible_change else None}
    elif scan_type == "Bearish (Downward Momentum)" and sma is not None and last_price < sma and rsi is not None and rsi > 30:
        return {"Symbol": symbol, "Last Price": last_price, "SMA": round(sma, 2), "RSI": round(rsi, 2), "Near Support": near_support, "Near Resistance": near_resistance, "Volume": current_volume, "Breakout Type": breakout_type, "Possible Change (%)": round(possible_change, 2) if possible_change else None}
    elif scan_type == "Breakouts":
        if breakout_type:
            return {"Symbol": symbol, "Breakout Type": breakout_type, "Last Price": last_price, "Recent High": recent_high, "Recent Low": recent_low, "Volume": current_volume, "Possible Change (%)": round(possible_change, 2) if possible_change else None}
        elif near_support or near_resistance:
            return {"Symbol": symbol, "Potential Breakout": "Support" if near_support else "Resistance", "Last Price": last_price, "Recent High": recent_high, "Recent Low": recent_low, "Volume": current_volume, "Possible Change (%)": round(possible_change, 2) if possible_change else None}
    elif scan_type == "Unusual Volume" and current_volume > volume_threshold * avg_volume:
        return {"Symbol": symbol, "Volume": current_volume, "Avg Volume": avg_volume, "Last Price": last_price}
    return None

def get_financial_metrics(symbol: str) -> Dict[str, float]:
    try:
        income_statement = requests.get(f"{FMP_BASE_URL}/income-statement/{symbol}?apikey={FMP_API_KEY}").json()
        balance_sheet = requests.get(f"{FMP_BASE_URL}/balance-sheet-statement/{symbol}?apikey={FMP_API_KEY}").json()
        cash_flow = requests.get(f"{FMP_BASE_URL}/cash-flow-statement/{symbol}?apikey={FMP_API_KEY}").json()
        key_metrics = requests.get(f"{FMP_BASE_URL}/key-metrics/{symbol}?apikey={FMP_API_KEY}").json()
        quote = requests.get(f"{FMP_BASE_URL}/quote/{symbol}?apikey={FMP_API_KEY}").json()
        if not income_statement or not balance_sheet or not cash_flow or not key_metrics or not quote:
            return {}
        latest_income = income_statement[0] if income_statement else {}
        latest_balance = balance_sheet[0] if balance_sheet else {}
        latest_cash_flow = cash_flow[0] if cash_flow else {}
        latest_metrics = key_metrics[0] if key_metrics else {}
        latest_quote = quote[0] if quote else {}
        return {
            "Current Price": latest_quote.get("price", 0), "EBITDA": latest_income.get("ebitda", 0), "Revenue": latest_income.get("revenue", 0),
            "Net Income": latest_income.get("netIncome", 0), "ROA": latest_metrics.get("roa", 0), "ROE": latest_metrics.get("roe", 0),
            "Beta": latest_metrics.get("beta", 0), "PE Ratio": latest_metrics.get("peRatio", 0), "Debt-to-Equity Ratio": latest_metrics.get("debtToEquity", 0),
            "Working Capital": latest_balance.get("totalCurrentAssets", 0) - latest_balance.get("totalCurrentLiabilities", 0),
            "Total Assets": latest_balance.get("totalAssets", 0), "Retained Earnings": latest_balance.get("retainedEarnings", 0),
            "EBIT": latest_income.get("ebit", 0), "Market Cap": latest_metrics.get("marketCap", 0), "Total Liabilities": latest_balance.get("totalLiabilities", 0),
            "Operating Cash Flow": latest_cash_flow.get("operatingCashFlow", 0), "Current Ratio": latest_metrics.get("currentRatio", 0),
            "Long Term Debt": latest_balance.get("longTermDebt", 0), "Shares Outstanding": latest_metrics.get("sharesOutstanding", 0),
            "Gross Margin": latest_metrics.get("grossProfitMargin", 0), "Asset Turnover": latest_metrics.get("assetTurnover", 0),
            "Capital Expenditure": latest_cash_flow.get("capitalExpenditure", 0), "Free Cash Flow": latest_cash_flow.get("freeCashFlow", 0),
            "Weighted Average Shares Diluted": latest_income.get("weightedAverageShsOutDil", 0), "Property Plant Equipment Net": latest_balance.get("propertyPlantEquipmentNet", 0),
            "Cash and Cash Equivalents": latest_balance.get("cashAndCashEquivalents", 0), "Total Debt": latest_balance.get("totalDebt", 0),
            "Interest Expense": latest_income.get("interestExpense", 0), "Short Term Debt": latest_balance.get("shortTermDebt", 0),
            "Intangible Assets": latest_balance.get("intangibleAssets", 0), "Accounts Receivable": latest_balance.get("accountsReceivable", 0),
            "Inventory": latest_balance.get("inventory", 0), "Accounts Payable": latest_balance.get("accountsPayable", 0),
            "COGS": latest_income.get("costOfRevenue", 0), "Tax Rate": latest_income.get("incomeTaxExpense", 0) / latest_income.get("incomeBeforeTax", 1) if latest_income.get("incomeBeforeTax", 1) != 0 else 0
        }
    except Exception as e:
        st.warning(f"⚠️ Error fetching financial metrics for {symbol}: {str(e)}")
        return {}

def get_historical_prices_fmp(symbol: str, period: str = "daily", limit: int = 30) -> (List[float], List[int]):
    try:
        response = requests.get(f"{FMP_BASE_URL}/historical-price-full/{symbol}?apikey={FMP_API_KEY}&timeseries={limit}")
        response.raise_for_status()
        data = response.json()
        if not data or "historical" not in data:
            return [], []
        prices = [day["close"] for day in data["historical"]]
        volumes = [day["volume"] for day in data["historical"]]
        return prices, volumes
    except Exception as e:
        st.warning(f"⚠️ Error fetching historical data for {symbol}: {str(e)}")
        return [], []

def speculate_next_day_movement(metrics: Dict[str, float], prices: List[float], volumes: List[int]) -> (str, float, Optional[float]):
    sma = calculate_sma(prices, period=50)
    rsi = calculate_rsi(prices, period=14)
    recent_high = max(prices[-10:]) if len(prices) >= 10 else None
    recent_low = min(prices[-10:]) if len(prices) >= 10 else None
    last_price = prices[-1] if prices else None
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else None
    current_volume = volumes[-1] if volumes else None
    trend = "High Volatility"
    confidence = 0.5
    if last_price is not None and sma is not None and rsi is not None:
        if rsi < 30 and last_price < sma:
            trend = "Bearish"
            confidence = 0.7 if current_volume and avg_volume and current_volume > avg_volume else 0.6
        elif rsi > 70 and last_price > sma:
            trend = "Bullish"
            confidence = 0.8 if current_volume and avg_volume and current_volume > avg_volume else 0.7
        elif recent_high and last_price > recent_high:
            trend = "Breakout (Bullish)"
            confidence = 0.9
        elif recent_low and last_price < recent_low:
            trend = "Breakdown (Bearish)"
            confidence = 0.9
    if metrics.get("ROE", 0) > 0.15 and metrics.get("Free Cash Flow", 0) > 0:
        confidence += 0.1
    if metrics.get("Current Ratio", 0) < 1:
        confidence -= 0.1
    if metrics.get("Beta", 0) > 1.5:
        confidence += 0.1 if trend == "Bullish" else -0.1
    predicted_change = (last_price * 0.01) * confidence if trend == "Bullish" else -(last_price * 0.01) * confidence
    predicted_price = last_price + predicted_change if last_price is not None else None
    return trend, confidence, predicted_price

def get_option_data(symbol: str, expiration_date: str) -> pd.DataFrame:
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {"symbol": symbol, "expiration": expiration_date, "greeks": "true"}
    response = requests.get(url, headers=HEADERS_TRADIER, params=params)
    if response.status_code != 200:
        st.error(f"Error al obtener los datos de opciones. Código de estado: {response.status_code}")
        return pd.DataFrame()
    data = response.json()
    if 'options' in data and 'option' in data['options']:
        options = data['options']['option']
        df = pd.DataFrame(options)
        df['action'] = df.apply(lambda row: "buy" if (row.get("bid", 0) > 0 and row.get("ask", 0) > 0) else "sell", axis=1)
        return df
    st.error("No se encontraron datos de opciones en la respuesta de la API.")
    return pd.DataFrame()

def fetch_data(endpoint: str, ticker: str = None, additional_params: dict = None):
    url = f"{FMP_BASE_URL}/{endpoint}"
    params = {"apikey": FMP_API_KEY}
    if ticker:
        params["symbol"] = ticker
    if additional_params:
        params.update(additional_params)
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) == 0:
                st.warning(f"No se encontraron datos para el endpoint: {endpoint}")
                return None
            return data
        st.error(f"Error al obtener datos: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        st.error(f"Error en la solicitud HTTP: {str(e)}")
        return None

def get_institutional_holders_list(ticker: str):
    endpoint = f"institutional-holder/{ticker}"
    data = fetch_data(endpoint, ticker)
    if data:
        return pd.DataFrame(data)
    return None

def estimate_greeks(strike: float, current_price: float, days_to_expiration: int, iv: float, option_type: str) -> Dict[str, float]:
    t = days_to_expiration / 365.0
    if iv <= 0 or t <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
    s = current_price
    k = strike
    r = RISK_FREE_RATE
    sigma = iv
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == "CALL":
        delta = norm.cdf(d1)
        theta = (-s * norm.pdf(d1) * sigma / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d2)) / 365.0
    else:
        delta = norm.cdf(d1) - 1
        theta = (-s * norm.pdf(d1) * sigma / (2 * np.sqrt(t)) + r * k * np.exp(-r * t) * norm.cdf(-d2)) / 365.0
    gamma = norm.pdf(d1) / (s * sigma * np.sqrt(t))
    vega = s * norm.pdf(d1) * np.sqrt(t) / 100.0
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}

def analyze_options(options_data: List[Dict], current_price: float) -> Dict[str, Dict[float, Dict[str, float]]]:
    analysis = {"CALL": {}, "PUT": {}}
    if not options_data:
        logger.warning("No options data to analyze")
        return analysis
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    exp_date = datetime.strptime(options_data[0].get("expiration_date") or options_data[0].get("expirationDate"), "%Y-%m-%d")
    days_to_exp = (exp_date - today).days
    
    for option in options_data:
        try:
            strike = float(option["strike"])
            option_type = option["option_type"].upper() if "option_type" in option else ("CALL" if option.get("type") == "call" else "PUT")
            bid_ask_spread = float(option.get('ask', 0)) - float(option.get('bid', 0))
            iv = float(option.get('implied_volatility', 0) or option.get('impliedVolatility', 0) or 0)
            volume = int(option.get('volume', 0) or 0)
            open_interest = int(option.get('open_interest', 0) or option.get('openInterest', 0) or 0)
            intrinsic = max(current_price - strike, 0) if option_type == "CALL" else max(strike - current_price, 0)
            greek = option.get("greeks", {})
            
            if greek and all(greek.get(k) is not None and greek.get(k) != 0 for k in ['delta', 'gamma']):
                delta = float(greek.get('delta', 0))
                gamma = float(greek.get('gamma', 0))
                theta = float(greek.get('theta', 0))
                vega = float(greek.get('vega', 0))
            else:
                estimated = estimate_greeks(strike, current_price, days_to_exp, iv if iv > 0 else 0.2, option_type)
                delta = estimated['delta']
                gamma = estimated['gamma']
                theta = estimated['theta']
                vega = estimated['vega']
            
            if strike not in analysis[option_type]:
                analysis[option_type][strike] = {
                    'gamma': gamma,
                    'vega': vega,
                    'theta': theta,
                    'delta': delta,
                    'iv': iv if iv > 0 else 0.2,
                    'bid': float(option.get('bid', 0)),
                    'ask': float(option.get('ask', 0)),
                    'spread': bid_ask_spread,
                    'open_interest': open_interest,
                    'volume': volume,
                    'intrinsic': intrinsic
                }
        except (ValueError, TypeError) as e:
            logger.error(f"Error analyzing option: {e} - {option}")
    logger.info(f"Analyzed: {len(analysis['CALL'])} CALLs, {len(analysis['PUT'])} PUTs")
    return analysis

def calculate_special_monetization(data: Dict, current_price: float, days_to_expiration: int) -> Tuple[float, float, float, str]:
    strike = list(data.keys())[0]
    option_type = 'CALL' if data[strike]['delta'] > 0 else 'PUT'
    mid_price = (data[strike]['bid'] + data[strike]['ask']) / 2
    delta = abs(data[strike]['delta'])
    gamma = data[strike]['gamma']
    theta = data[strike]['theta']
    iv = data[strike]['iv']
    intrinsic = data[strike]['intrinsic']
    open_interest = data[strike]['open_interest']
    
    gamma_iv_index = gamma * iv * (open_interest / 1000000.0) if gamma > 0 and iv > 0 else 0.001
    t = days_to_expiration / 365.0
    d1 = (np.log(current_price / strike) + (RISK_FREE_RATE + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    prob_otm = norm.cdf(-d1) if option_type == "PUT" else norm.cdf(d1)
    
    direction_factor = 1 if (option_type == "CALL" and current_price > strike) or (option_type == "PUT" and current_price < strike) else 0.5
    monetization_factor = mid_price * (1 + abs(theta) / (gamma + 0.001)) * direction_factor
    
    potential_profit = monetization_factor * 100
    risk = mid_price * 100 * (1 - prob_otm) * (1 + gamma * 5)
    rr_ratio = potential_profit / risk if risk > 0 else 10.0
    action = "SELL" if prob_otm > 0.5 else "BUY"
    
    logger.debug(f"Strike {strike}: Gamma-IV Index={gamma_iv_index:.4f}, RR={rr_ratio:.2f}, Prob OTM={prob_otm:.2%}, Mid Price={mid_price}, Profit={potential_profit:.2f}, Open Interest={open_interest}")
    return rr_ratio, potential_profit, prob_otm, action

def generate_contract_suggestions(ticker: str, options_data: List[Dict], current_price: float, open_interest_threshold: int, gamma_threshold: float) -> List[Dict]:
    if not options_data or not current_price:
        logger.error("No options data or invalid price")
        return []
    
    exp_date = datetime.strptime(options_data[0].get("expiration_date") or options_data[0].get("expirationDate"), "%Y-%m-%d")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    days_to_expiration = (exp_date - today).days
    if days_to_expiration < 0:
        logger.error(f"Expiration date {exp_date} is in the past")
        return []
    
    options_analysis = analyze_options(options_data, current_price)
    if not options_analysis["CALL"] and not options_analysis["PUT"]:
        return []
    
    max_pain_strike = calculate_max_pain_optimized(options_data)
    
    suggestions = []
    for option_type in ["CALL", "PUT"]:
        strikes = sorted(options_analysis[option_type].keys())
        relevant_strikes = [s for s in strikes if (option_type == "CALL" and s > current_price) or (option_type == "PUT" and s < current_price)]
        
        if not relevant_strikes:
            logger.warning(f"No OTM strikes for {option_type}")
        
        for strike in relevant_strikes:
            data = options_analysis[option_type][strike]
            open_interest = data['open_interest']
            gamma = data['gamma']
            if open_interest >= open_interest_threshold and gamma >= gamma_threshold:
                rr_ratio, profit, prob_otm, action = calculate_special_monetization({strike: data}, current_price, days_to_expiration)
                vol_category = "HighOpenInterest"
                reason = f"{vol_category}: Strike {strike}, Gamma {data['gamma']:.4f}, IV {data['iv']:.2f}, Delta {data['delta']:.2f}, RR {rr_ratio:.2f}, Prob OTM {prob_otm:.2%}, Profit ${profit:.2f}, OI {open_interest}"
                suggestions.append({
                    "Action": action,
                    "Type": option_type,
                    "Strike": strike,
                    "Reason": reason,
                    "Gamma": data['gamma'],
                    "IV": data['iv'],
                    "Delta": data['delta'],
                    "RR": rr_ratio,
                    "Prob OTM": prob_otm,
                    "Profit": profit,
                    "Open Interest": open_interest,
                    "IsMaxPain": strike == max_pain_strike
                })
                logger.info(f"Added {option_type} strike {strike}: Open Interest={open_interest}, Gamma={data['gamma']:.4f}, IV={data['iv']:.2f}, Max Pain={strike == max_pain_strike}")
    
    if max_pain_strike:
        for option_type in ["CALL", "PUT"]:
            if max_pain_strike in options_analysis[option_type]:
                data = options_analysis[option_type][max_pain_strike]
                rr_ratio, profit, prob_otm, action = calculate_special_monetization({max_pain_strike: data}, current_price, days_to_expiration)
                reason = f"MaxPain: Strike {max_pain_strike}, Gamma {data['gamma']:.4f}, IV {data['iv']:.2f}, Delta {data['delta']:.2f}, RR {rr_ratio:.2f}, Prob OTM {prob_otm:.2%}, Profit ${profit:.2f}, OI {data['open_interest']}"
                if not any(s["Strike"] == max_pain_strike and s["Type"] == option_type for s in suggestions):
                    suggestions.append({
                        "Action": action,
                        "Type": option_type,
                        "Strike": max_pain_strike,
                        "Reason": reason,
                        "Gamma": data['gamma'],
                        "IV": data['iv'],
                        "Delta": data['delta'],
                        "RR": rr_ratio,
                        "Prob OTM": prob_otm,
                        "Profit": profit,
                        "Open Interest": data['open_interest'],
                        "IsMaxPain": True
                    })
                    logger.info(f"Added Max Pain {option_type} strike {max_pain_strike}")

    logger.info(f"Generated {len(suggestions)} suggestions for {exp_date.strftime('%Y-%m-%d')} with OI >= {open_interest_threshold}, Gamma >= {gamma_threshold}")
    return suggestions


# --- Main App ---

# --- Main App ---
def main():
    st.set_page_config(page_title="O Z Y |  DATA®", page_icon="♾️", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .stApp {background-color: #1E1E1E;}
        .stTextInput, .stSelectbox {background-color: #2D2D2D; color: #FFFFFF;}
        .stSpinner > div > div {border-color: #32CD32 !important;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
      PRO SCANNER |®
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Options Scanner", "Market Scanner", "News", "Institutional Holders", "Stock Analysis", "Trading Options"])

    with tab1:
        st.subheader("Options Scanner")
        ticker = st.text_input("Ticker", value="SPY", key="ticker_input").upper()
        expiration_dates = get_expiration_dates(ticker)
        if not expiration_dates:
            st.error(f"What were you thinking, '{ticker}'? You're a trader and you mess this up? If you trade like this, you're doomed!")
            return
        expiration_date = st.selectbox("Expiration Date", expiration_dates, key="expiration_date")
        with st.spinner("Fetching price..."):
            current_price = get_current_price(ticker)
            if current_price == 0.0:
                st.error(f"Invalid ticker '{ticker}' or no price data available.")
                return
        st.markdown(f"**Current Price:** ${current_price:.2f}")

        with st.spinner(f"Fetching data for {expiration_date}..."):
            options_data = get_options_data(ticker, expiration_date)
            if not options_data:
                st.error("No options data available for this ticker and expiration date.")
                return
            processed_data = {}
            for opt in options_data:
                if not opt or not isinstance(opt, dict):
                    continue
                strike = float(opt.get("strike", 0))
                option_type = opt.get("option_type", "").upper()
                if option_type not in ["CALL", "PUT"]:
                    continue
                oi = int(opt.get("open_interest", 0))
                greeks = opt.get("greeks", {})
                gamma = float(greeks.get("gamma", 0)) if isinstance(greeks, dict) else 0
                if strike not in processed_data:
                    processed_data[strike] = {"CALL": {"OI": 0, "Gamma": 0}, "PUT": {"OI": 0, "Gamma": 0}}
                processed_data[strike][option_type]["OI"] += oi
                processed_data[strike][option_type]["Gamma"] += gamma
            if not processed_data:
                st.error("No valid data to display.")
                return
            prices, _ = get_historical_prices_combined(ticker)
            historical_prices = prices
            touched_strikes = detect_touched_strikes(processed_data.keys(), historical_prices)
            max_pain = calculate_max_pain_optimized(options_data)
            df = analyze_contracts(ticker, expiration_date, current_price)
            max_pain_strike, max_pain_df = calculate_max_pain(df)
            st.plotly_chart(gamma_exposure_chart(processed_data, current_price, touched_strikes), use_container_width=True)
            skew_fig, total_calls, total_puts = plot_skew_analysis_with_totals(options_data, current_price)
            st.plotly_chart(skew_fig, use_container_width=True)
            st.write(f"**Total CALLS:** {total_calls} | **Total PUTS:** {total_puts}")
            st.write(f"Current Price of {ticker}: ${current_price:.2f} (Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            st.write(f"**Max Pain Strike (Optimized):** {max_pain if max_pain else 'N/A'}")
            st.plotly_chart(plot_max_pain_histogram_with_levels(max_pain_df, current_price), use_container_width=True)

    with tab2:
        st.subheader("Market Scanner")
        scan_type = st.selectbox("Select Scan Type", ["Bullish (Upward Momentum)", "Bearish (Downward Momentum)", "Breakouts", "Unusual Volume"])
        max_results = st.slider("Max Stocks to Display", 1, 200, 30)
        
        if st.button("🚀 Start Batch Scan"):
            with st.spinner("Scanning market..."):
                stock_list = get_stock_list_combined()
                if not stock_list:
                    st.error("No se pudo obtener la lista de acciones.")
                    return

                num_workers = min(50, max(10, len(stock_list) // 5))
                results = []
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(scan_stock, symbol, scan_type) for symbol in stock_list]
                    for future in futures:
                        result = future.result()
                        if result:
                            results.append(result)
                
                if results:
                    df_results = pd.DataFrame(results[:max_results])
                    styled_df = df_results.style.background_gradient(cmap="Blues").set_properties(**{"text-align": "center"})
                    st.dataframe(styled_df, use_container_width=True)
                    
                    if "Volume" in df_results.columns:
                        fig = go.Figure()
                        for _, row in df_results.iterrows():
                            color = "green" if row.get("Breakout Type") == "Up" else "red" if row.get("Breakout Type") == "Down" else "blue"
                            fig.add_trace(go.Bar(x=[row["Symbol"]], y=[row["Volume"]], marker_color=color,
                                                 hovertext=f"Symbol: {row['Symbol']}<br>Volume: {row['Volume']:,}<br>Breakout: {row.get('Breakout Type', 'N/A')}<br>Change: {row.get('Possible Change (%)', 'N/A')}%"))
                        fig.update_layout(title="📊 Volume Distribution", xaxis_title="Stock Symbol", yaxis_title="Volume", template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No stocks match the criteria.")
                
                if results:
                    csv = pd.DataFrame(results).to_csv(index=False)
                    st.download_button(label="📥 Export Results to CSV", data=csv, file_name="market_scan_results.csv", mime="text/csv")

    with tab3:
        st.subheader("News Scanner")
        keywords = st.text_input("Enter keywords (comma-separated):", "Trump").split(",")
        keywords = [k.strip() for k in keywords if k.strip()]
        if st.button("Fetch News"):
            with st.spinner("Fetching news..."):
                google_news = fetch_google_news(keywords)
                bing_news = fetch_bing_news(keywords)
                instagram_posts = fetch_instagram_posts(keywords)
                latest_news = google_news + bing_news + instagram_posts
                if latest_news:
                    for idx, article in enumerate(latest_news[:10], 1):
                        st.markdown(f"### {idx}. [{article['title']}]({article['link']})")
                        st.markdown(f"**Published:** {article['time']}\n")
                        st.markdown("---")
                else:
                    st.error("No recent news found.")

    with tab4:
        st.subheader("Institutional Holders")
        ticker = st.text_input("Ticker for Holders (e.g., AAPL):", "AAPL", key="holders_ticker").upper()
        if ticker:
            holders = get_institutional_holders_list(ticker)
            if holders is not None and not holders.empty:
                st.dataframe(holders)
            else:
                st.warning("No institutional holders data available.")

    with tab5:
        st.subheader("Stock Analysis")
        stock = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="SPY", key="stock_analysis").upper()
        expiration_dates = get_expiration_dates(stock)
        if not expiration_dates:
            st.error(f"No expiration dates found for '{stock}'. Please enter a valid stock ticker (e.g., SPY, AAPL).")
            return
        selected_expiration = st.selectbox("Select an Expiration Date:", expiration_dates, key="stock_exp_date")
        if stock:
            with st.spinner("Fetching data..."):
                financial_metrics = get_financial_metrics(stock)
                prices, volumes = get_historical_prices_fmp(stock)
                if not prices or not volumes:
                    st.error(f"❌ Unable to fetch data for {stock}.")
                else:
                    trend, confidence, predicted_price = speculate_next_day_movement(financial_metrics, prices, volumes)
                    current_price = get_current_price(stock)
                    if current_price == 0.0:
                        st.error(f"❌ No se pudo obtener el precio actual para {stock}. Verifica el ticker o la conexión a la API.")
                        return
                    st.markdown(f"### Metrics for {stock}")
                    st.write(f"- **Current Price**: ${current_price:,.2f}")
                    st.write(f"- **EBITDA**: ${financial_metrics.get('EBITDA', 0):,.2f}")
                    st.write(f"- **Revenue**: ${financial_metrics.get('Revenue', 0):,.2f}")
                    st.write(f"- **Net Income**: ${financial_metrics.get('Net Income', 0):,.2f}")
                    st.write(f"- **ROE**: {financial_metrics.get('ROE', 0):.2f}")
                    st.write(f"- **Beta**: ${financial_metrics.get('Beta', 0):,.2f}")
                    st.write(f"- **PE Ratio**: ${financial_metrics.get('PE Ratio', 0):,.2f}")
                    st.write(f"- **Debt-to-Equity Ratio**: {financial_metrics.get('Debt-to-Equity Ratio', 0):.2f}")
                    st.write(f"- **Market Cap**: ${financial_metrics.get('Market Cap', 0):,.2f}")
                    st.write(f"- **Operating Cash Flow**: ${financial_metrics.get('Operating Cash Flow', 0):,.2f}")
                    st.write(f"- **Free Cash Flow**: ${financial_metrics.get('Free Cash Flow', 0):,.2f}")
                    st.markdown(f"### Speculation for {stock}")
                    st.write(f"- **Trend**: {trend}")
                    st.write(f"- **Confidence**: {confidence:.2f}")
                    st.write(f"- **Predicted Price (Next Day)**: ${predicted_price:.2f}" if predicted_price is not None else "- **Predicted Price (Next Day)**: N/A")
                    option_data = get_option_data(stock, selected_expiration)
                    if not option_data.empty:
                        option_data_list = option_data.to_dict('records')
                        buy_calls = option_data[(option_data["option_type"] == "call") & (option_data["action"] == "buy")]
                        sell_calls = option_data[(option_data["option_type"] == "call") & (option_data["action"] == "sell")]
                        buy_puts = option_data[(option_data["option_type"] == "put") & (option_data["action"] == "buy")]
                        sell_puts = option_data[(option_data["option_type"] == "put") & (option_data["action"] == "sell")]
                        all_strikes = sorted(set(option_data["strike"]))
                        buy_calls_data = pd.DataFrame({"strike": all_strikes}).merge(buy_calls[["strike", "open_interest"]], on="strike", how="left").fillna({"open_interest": 0})
                        sell_calls_data = pd.DataFrame({"strike": all_strikes}).merge(sell_calls[["strike", "open_interest"]], on="strike", how="left").fillna({"open_interest": 0})
                        buy_puts_data = pd.DataFrame({"strike": all_strikes}).merge(buy_puts[["strike", "open_interest"]], on="strike", how="left").fillna({"open_interest": 0})
                        sell_puts_data = pd.DataFrame({"strike": all_strikes}).merge(sell_puts[["strike", "open_interest"]], on="strike", how="left").fillna({"open_interest": 0})
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name="Buy Call", x=buy_calls_data["strike"], y=buy_calls_data["open_interest"],
                                             marker_color="green", text="Buy Call", textposition="inside"))
                        fig.add_trace(go.Bar(name="Sell Call", x=sell_calls_data["strike"], y=sell_calls_data["open_interest"],
                                             marker_color="orange", text="Sell Call", textposition="inside"))
                        fig.add_trace(go.Bar(name="Buy Put", x=buy_puts_data["strike"], y=-buy_puts_data["open_interest"],
                                             marker_color="red", text="Buy Put", textposition="inside"))
                        fig.add_trace(go.Bar(name="Sell Put", x=sell_puts_data["strike"], y=-sell_puts_data["open_interest"],
                                             marker_color="purple", text="Sell Put", textposition="inside"))

                        # Cálculo combinado ajustado para la dirección del MM
                        max_pain = calculate_max_pain_optimized(option_data_list)
                        total_call_oi = sum(row["open_interest"] for row in option_data_list if row["option_type"] == "call" and row["strike"] > current_price)
                        total_put_oi = sum(row["open_interest"] for row in option_data_list if row["option_type"] == "put" and row["strike"] < current_price)
                        total_oi = total_call_oi + total_put_oi
                        gamma_calls = sum(row["greeks"]["gamma"] * row["open_interest"] if isinstance(row["greeks"], dict) and "gamma" in row["greeks"] else 0 
                                          for row in option_data_list if row["option_type"] == "call" and "greeks" in row)
                        gamma_puts = sum(row["greeks"]["gamma"] * row["open_interest"] if isinstance(row["greeks"], dict) and "gamma" in row["greeks"] else 0 
                                         for row in option_data_list if row["option_type"] == "put" and "greeks" in row)
                        net_gamma = gamma_calls - gamma_puts
                        max_pain_factor = -2 if current_price > max_pain else 2 if current_price < max_pain else 0
                        oi_pressure = (total_call_oi - total_put_oi) / max(total_oi, 1)
                        gamma_factor = net_gamma / 10000
                        combined_score = max_pain_factor + oi_pressure + gamma_factor
                        direction_mm = "Down" if combined_score < 0 else "Up" if combined_score > 0 else "Neutral"
                        st.write(f"Debug: Current Price = {current_price}, Max Pain = {max_pain}, OI Pressure = {oi_pressure}, Net Gamma = {net_gamma}, Combined Score = {combined_score}, Direction = {direction_mm}")

                        # Añadir línea vertical para Current Price (delgada, con label transparente)
                        y_max = max(buy_calls_data["open_interest"].max() or 0, sell_calls_data["open_interest"].max() or 0) * 1.1 or 100
                        fig.add_trace(go.Scatter(x=[current_price, current_price], y=[-y_max, y_max], mode="lines",
                                                 line=dict(width=1, dash="dash", color="yellow"),
                                                 name="Current Price",
                                                 hovertemplate=f"Current Price: ${current_price:.2f}<br>",
                                                 showlegend=False))

                        # Añadir anotación dinámica para la dirección del MM con colores
                        score_color = "red" if combined_score < 0 else "green" if combined_score > 0 else "white"
                        fig.add_annotation(x=current_price, y=y_max * 0.9,
                                          text=f"MM Score: <span style='color:{score_color}'>{combined_score:.2f}</span><br>{direction_mm}",
                                          showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="yellow",
                                          font=dict(size=12), ax=20, ay=-30, bgcolor="rgba(0,0,0,0.5)", bordercolor="yellow")

                        # Ajustar layout para tamaño grande
                        fig.update_layout(title=f"Calls and Puts for {stock} (Expiration: {selected_expiration})",
                                          xaxis_title="Strike Price", yaxis_title="Open Interest", barmode="relative",
                                          showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                                          xaxis=dict(range=[min(all_strikes) - 10, max(all_strikes) + 10]),
                                          height=600)

                        # Mostrar gráfica con tamaño completo
                        st.plotly_chart(fig, use_container_width=True, height=600)

    with tab6:
        st.subheader("Trading Options")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### Configuration")
            ticker = st.text_input("Ticker Symbol (e.g., SPY)", "SPY", key="alerts_ticker").upper()
            expiration_dates = get_expiration_dates(ticker)
            if not expiration_dates:
                st.error(f"What were you thinking, '{ticker}'? You're a trader and you mess this up? If you trade like this, you're doomed!")
                return
            expiration_date = st.selectbox("Expiration Date", expiration_dates, key="alerts_exp_date")
            with st.spinner("Fetching price..."):
                current_price = get_current_price(ticker)
                if current_price == 0.0:
                    st.error(f"Invalid ticker '{ticker}' or no price data available.")
                    return
            st.markdown("### Vol Filter")
            volume_options = {
                "0.1M": 10000,
                "0.2M": 20000,
                "0.3M": 30000,
                "0.4M": 40000,
                "0.5M": 50000,
                "1.0M": 100000
            }
            selected_volume = st.selectbox("Min Open Interest (M)", list(volume_options.keys()), index=3, key="alerts_vol")
            open_interest_threshold = volume_options[selected_volume]
            st.markdown("### Gamma Filter")
            gamma_options = {
                "0.001": 0.001,
                "0.01": 0.01,
                "0.02": 0.02,
                "0.03": 0.03,
                "0.04": 0.04,
                "0.05": 0.05
            }
            selected_gamma = st.selectbox("Min Gamma", list(gamma_options.keys()), index=2, key="alerts_gamma")
            gamma_threshold = gamma_options[selected_gamma]
            st.markdown(f"**Current Price:** ${current_price:.2f}  \n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        with col2:
            st.markdown(f"**Current Price:** ${current_price:.2f}  \n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            with st.spinner(f"Generating alerts for {expiration_date}..."):
                options_data = get_options_data(ticker, expiration_date)
                if not options_data:
                    st.error("No options data available for this date.")
                    return
                suggestions = generate_contract_suggestions(ticker, options_data, current_price, open_interest_threshold, gamma_threshold)
                if suggestions:
                    df = pd.DataFrame(suggestions)
                    df['Contract'] = df.apply(lambda row: f"{ticker} {row['Action']} {row['Type']} {row['Strike']}", axis=1)
                    df = df[['Contract', 'Strike', 'Action', 'Type', 'Gamma', 'IV', 'Delta', 'RR', 'Prob OTM', 'Profit', 'Open Interest', 'IsMaxPain']]
                    df.columns = ['Contract', 'Strike', 'Action', 'Type', 'Gamma', 'IV', 'Delta', 'R/R', 'Prob OTM', 'Profit ($)', 'Open Int.', 'Max Pain']
                    def color_row(row):
                        if row['Max Pain']:
                            return ['color: #FFA500'] * len(row)
                        elif row['Type'] == "CALL":
                            return ['color: #32CD32' if row['Action'] == "SELL" and row['Strike'] > current_price else 'color: #008000'] * len(row)
                        elif row['Type'] == "PUT":
                            return ['color: #FF4500' if row['Action'] == "SELL" and row['Strike'] < current_price else 'color: #FF0000'] * len(row)
                        return [''] * len(row)
                    styled_df = df.style.apply(color_row, axis=1).format({
                        'Strike': '{:.1f}',
                        'Gamma': '{:.4f}',
                        'IV': '{:.2f}',
                        'Delta': '{:.2f}',
                        'R/R': '{:.2f}',
                        'Prob OTM': '{:.2%}',
                        'Profit ($)': '{:.2f}',
                        'Open Int.': '{:,.0f}'
                    })
                    st.dataframe(styled_df, height=400)
                else:
                    st.error(f"No alerts generated with Open Interest ≥ {selected_volume}, Gamma ≥ {selected_gamma}. Check logs.")

    st.markdown("---")
    st.markdown("*Developed by Ozy | © 2025*")

if __name__ == "__main__":
    main()
