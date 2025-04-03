import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import List, Dict, Optional
import plotly.graph_objects as go
from datetime import datetime, timedelta
import multiprocessing
import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import csv
import bcrypt
import sqlite3
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
import streamlit.components.v1 as components
import math
import krakenex
import base64
import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)

# API Sessions and Configurations
session_fmp = requests.Session()
session_tradier = requests.Session()
retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy)
session_fmp.mount("https://", adapter)
session_tradier.mount("https://", adapter)
num_workers = min(100, multiprocessing.cpu_count() * 2)

# API Keys and Constants
# Configurar cliente de Kraken con las claves proporcionadas
API_KEY = "kyFpw+5fbrFIMDuWJmtkbbbr/CgH/MS63wv7dRz3rndamK/XnjNOVkgP"
PRIVATE_KEY = "7xbaBIp902rSBVdIvtfrUNbRHEHMkfMHPEf4rssz+ZwSwjUZFegjdyyYZzcE5DbBrUbtFdGRRGRjTuTnEblZWA=="
kraken = krakenex.API(key=API_KEY, secret=PRIVATE_KEY)

FMP_API_KEY = "bQ025fPNVrYcBN4KaExd1N3Xczyk44wM"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
TRADIER_API_KEY = "d0H5QGsma6Bh41VBw6P6lItCBl7D"
TRADIER_BASE_URL = "https://api.tradier.com/v1"
HEADERS_FMP = {"Accept": "application/json"}
HEADERS_TRADIER = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}
PASSWORDS_DB = "auth_data/passwords.db"
CACHE_TTL = 300

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración inicial de página ---
st.set_page_config(
    page_title="𝗢 𝗭 𝗬 |  DATA®",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuración de APIs ---
FMP_API_KEY = "bQ025fPNVrYcBN4KaExd1N3Xczyk44wM"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
TRADIER_API_KEY = "d0H5QGsma6Bh41VBw6P6lItCBl7D"
TRADIER_BASE_URL = "https://api.tradier.com/v1"

HEADERS_FMP = {"Accept": "application/json"}
HEADERS_TRADIER = {"Authorization": f"Bearer {TRADIER_API_KEY}", "Accept": "application/json"}

# --- Constantes ---
PASSWORDS_DB = "auth_data/passwords.db"
CACHE_TTL = 300
MAX_RETRIES = 5
INITIAL_DELAY = 1
RISK_FREE_RATE = 0.045

# --- Autenticación con SQLite ---
def initialize_passwords_db():
    os.makedirs("auth_data", exist_ok=True)
    conn = sqlite3.connect(PASSWORDS_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS passwords 
                 (password TEXT PRIMARY KEY, usage_count INTEGER DEFAULT 0, ip1 TEXT DEFAULT '', ip2 TEXT DEFAULT '')''')
    initial_passwords = [
        ("abc123", 0, "", ""),
        ("def456", 0, "", ""),
        ("ghi789", 0, "", ""),
        ("jkl010", 0, "", ""),
        ("mno345", 0, "", ""),
        ("pqr678", 0, "", ""),
        ("stu901", 0, "", ""),
        ("vwx234", 0, "", ""),
        ("yz1234", 0, "", ""),
        ("abcd56", 0, "", ""),
        ("efgh78", 0, "", ""),
        ("ijkl90", 0, "", ""),
        ("mnop12", 0, "", ""),
        ("qrst34", 0, "", ""),
        ("uvwx56", 0, "", ""),
        ("yzab78", 0, "", ""),
        ("cdef90", 0, "", ""),
        ("ghij12", 0, "", ""),
        ("news34", 0, "", ""),
        ("opqr56", 0, "", ""),
        ("xyz789", 0, "", ""),
        ("kml456", 0, "", ""),
        ("nop123", 0, "", ""),
        ("qwe987", 0, "", ""),
        ("asd654", 0, "", ""),
        ("zxc321", 0, "", ""),
        ("bnm098", 0, "", ""),
        ("vfr765", 0, "", ""),
        ("tgb432", 0, "", ""),
        ("hju109", 0, "", "")
    ]
    hashed_passwords = [(bcrypt.hashpw(pwd.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'), count, ip1, ip2) 
                        for pwd, count, ip1, ip2 in initial_passwords]
    c.executemany("INSERT OR IGNORE INTO passwords VALUES (?, ?, ?, ?)", hashed_passwords)
    conn.commit()
    conn.close()
    logger.info("Password database initialized with existing and new passwords.")

def load_passwords():
    conn = sqlite3.connect(PASSWORDS_DB)
    c = conn.cursor()
    c.execute("SELECT password, usage_count, ip1, ip2 FROM passwords")
    passwords = {row[0]: {"usage_count": row[1], "ip1": row[2], "ip2": row[3]} for row in c.fetchall()}
    conn.close()
    return passwords

def save_passwords(passwords):
    conn = sqlite3.connect(PASSWORDS_DB)
    c = conn.cursor()
    c.execute("DELETE FROM passwords")
    c.executemany("INSERT INTO passwords VALUES (?, ?, ?, ?)", 
                  [(pwd, data["usage_count"], data["ip1"], data["ip2"]) for pwd, data in passwords.items()])
    conn.commit()
    conn.close()
    logger.info("Passwords updated in database.")

def get_local_ip():
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        logger.error("Error obtaining local IP.")
        return None

def authenticate_password(input_password):
    local_ip = get_local_ip()
    if not local_ip:
        st.error("Could not obtain local IP.")
        logger.error("Failed to obtain local IP during authentication.")
        return False
    passwords = load_passwords()
    for hashed_pwd, data in passwords.items():
        if bcrypt.checkpw(input_password.encode('utf-8'), hashed_pwd.encode('utf-8')):
            if data["usage_count"] < 2:
                if data["ip1"] == "":
                    passwords[hashed_pwd]["ip1"] = local_ip
                elif data["ip2"] == "" and data["ip1"] != local_ip:
                    passwords[hashed_pwd]["ip2"] = local_ip
                passwords[hashed_pwd]["usage_count"] += 1
                save_passwords(passwords)
                logger.info(f"Authentication successful for {input_password} from IP: {local_ip}, usage count: {passwords[hashed_pwd]['usage_count']}")
                return True
            elif data["usage_count"] == 2 and (data["ip1"] == local_ip or data["ip2"] == local_ip):
                logger.info(f"Repeat authentication successful for {input_password} from IP: {local_ip}")
                return True
            else:
                st.error("❌ This password has already been used by two IPs. To get your own access to OzyTarget, text 'OzyTarget Access' to 678-978-9414.")
                logger.warning(f"Authentication attempt for {input_password} from IP {local_ip} rejected; already used from {data['ip1']} and {data['ip2']}")
                return False
    logger.info(f"Reached error message for {input_password} from IP: {local_ip}")  # Log extra
    st.error("❌ Incorrect password. If you don’t have access, text 'OzyTarget Access' to 678-978-9414 to purchase your subscription.")
    logger.warning(f"Authentication failed: Invalid password {input_password}")
    return False

# Pantalla de autenticación con logo
initialize_passwords_db()
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    logo_path = "assets/favicon.png"
    if not os.path.exists(logo_path):
        logo_path = "favicon.png"
    if os.path.exists(logo_path):
        st.markdown("<div style='display: flex; justify-content: center; align-items: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
        st.image(logo_path, width=150)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No 'favicon.png' found for login screen. Place it in 'C:/Users/urbin/TradingApp/' or 'assets/'.")
    
    st.title("🔒 VIP ACCESS")
    
    # Estilo personalizado para el mensaje de carga
    st.markdown("""
    <style>
    .loading-container {
        text-align: center;
        padding: 25px;
        background: linear-gradient(145deg, #1A1A1A, #2D2D2D);
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.7), inset 0 0 10px rgba(50, 205, 50, 0.1);
        border: 1px solid rgba(50, 205, 50, 0.3);
    }
    .loading-text {
        font-size: 26px;
        font-weight: 600;
        color: #32CD32;
        text-shadow: 0 0 12px rgba(50, 205, 50, 0.8);
        letter-spacing: 1px;
    }
    .spinner-pro {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(255, 255, 255, 0.2);
        border-top: 4px solid #32CD32;
        border-radius: 50%;
        animation: spin-pro 1s ease-in-out infinite;
        margin: 15px auto 0;
    }
    @keyframes spin-pro {
        0% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); border-top-color: #FFD700; }
        100% { transform: rotate(360deg); border-top-color: #32CD32; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    password = st.text_input("Enter your password", type="password")
    if st.button("LogIn"):
        if not password:
            st.error("❌ Please enter a password.")
        elif authenticate_password(password):
            st.session_state["authenticated"] = True
            # Mensaje sofisticado con animación profesional
            with st.empty():
                st.markdown("""
                <div class="loading-container">
                    <div class="loading-text">✅ ACCESS GRANTED</div>
                    <div class="spinner-pro"></div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1)  # Carga rápida de 1 segundo
            st.rerun()
    st.stop()
########################################################app
########################################################app













@st.cache_data(ttl=3600)
def fetch_logo_url(symbol: str) -> str:
    """Obtiene la URL del logo de Clearbit con un fallback como base64."""
    url = f"https://logo.clearbit.com/{symbol.lower()}.com"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            logger.info(f"Logo fetched for {symbol}")
            return f"data:image/png;base64,{base64.b64encode(response.content).decode('utf-8')}"
    except Exception as e:
        logger.warning(f"Failed to fetch logo for {symbol}: {e}")
    default_logo_path = "default_logo.png"
    if os.path.exists(default_logo_path):
        with open(default_logo_path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    logger.info(f"Using fallback logo for {symbol}")
    return "https://via.placeholder.com/100"

@st.cache_data(ttl=86400)
def get_top_traded_stocks() -> set:
    """Obtiene una lista de las acciones más operadas desde FMP."""
    url = f"{FMP_BASE_URL}/stock-screener"
    params = {
        "apikey": FMP_API_KEY,
        "marketCapMoreThan": 10_000_000_000,
        "volumeMoreThan": 1_000_000,
        "exchange": "NASDAQ,NYSE",
        "limit": 100
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        top_stocks = {stock["symbol"] for stock in data if stock.get("isActivelyTrading", True)}
        logger.info(f"Fetched {len(top_stocks)} top traded stocks")
        return top_stocks
    except Exception as e:
        logger.error(f"Error fetching top traded stocks: {e}")
        # Fallback básico
        return {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "WMT", "SPY"}

def group_by_day_of_week(events: List[Dict], start_date: datetime, end_date: datetime) -> Dict[str, List[Dict]]:
    """Agrupa eventos por día de la semana entre start_date y end_date."""
    grouped = {}
    for event in events:
        event_date = datetime.strptime(event["Date"], "%Y-%m-%d").date()
        if start_date <= event_date <= end_date:
            day_name = event_date.strftime("%A")
            date_str = event_date.strftime("%Y-%m-%d")
            key = f"{day_name} ({date_str})"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(event)
    sorted_grouped = dict(sorted(grouped.items(), key=lambda x: datetime.strptime(x[0].split("(")[1].strip(")"), "%Y-%m-%d")))
    logger.info(f"Grouped {len(events)} events into {len(sorted_grouped)} days")
    return sorted_grouped

@st.cache_data(ttl=86400)
def get_implied_volatility(symbol: str) -> Optional[float]:
    """Obtiene la volatilidad implícita promedio de opciones cercanas desde Tradier."""
    expiration_dates = get_expiration_dates(symbol)
    if not expiration_dates:
        logger.warning(f"No expiration dates for {symbol}")
        return None
    nearest_exp = expiration_dates[0]
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {"symbol": symbol, "expiration": nearest_exp, "greeks": "true"}
    try:
        response = requests.get(url, headers=HEADERS_TRADIER, params=params, timeout=5)
        response.raise_for_status()
        data = response.json().get("options", {}).get("option", [])
        ivs = [float(opt.get("implied_volatility", 0)) for opt in data if opt.get("implied_volatility")]
        if ivs:
            avg_iv = sum(ivs) / len(ivs)
            logger.info(f"Average IV for {symbol}: {avg_iv}")
            return avg_iv
        return None
    except Exception as e:
        logger.error(f"Error fetching IV for {symbol}: {e}")
        return None

@st.cache_data(ttl=86400)
def get_historical_earnings_movement(symbol: str) -> Optional[float]:
    """Obtiene el movimiento promedio histórico post-earnings desde FMP."""
    url = f"{FMP_BASE_URL}/historical/earning_calendar/{symbol}"
    params = {"apikey": FMP_API_KEY, "limit": 4}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if not data or not isinstance(data, list):
            return None
        price_url = f"{FMP_BASE_URL}/historical-price-full/{symbol}"
        price_data = requests.get(price_url, params={"apikey": FMP_API_KEY, "timeseries": 10}).json().get("historical", [])
        movements = []
        for earning in data:
            earning_date = earning.get("date")
            if not earning_date:
                continue
            earning_date_obj = datetime.strptime(earning_date, "%Y-%m-%d").date()
            for i, price in enumerate(price_data):
                price_date = datetime.strptime(price["date"], "%Y-%m-%d").date()
                if price_date >= earning_date_obj and i > 0:
                    prev_close = price_data[i-1]["close"]
                    post_close = price["close"]
                    movement = abs((post_close - prev_close) / prev_close * 100)
                    movements.append(movement)
                    break
        if movements:
            avg_movement = sum(movements) / len(movements)
            logger.info(f"Historical earnings movement for {symbol}: {avg_movement}%")
            return avg_movement
        return None
    except Exception as e:
        logger.error(f"Error fetching historical earnings for {symbol}: {e}")
        return None

def calculate_possible_movement(symbol, eps_est, revenue_est, time):
    """Calcula el Possible Movement con precisión cercana a cero."""
    iv = get_implied_volatility(symbol)
    if iv is None:
        iv = 0.3  # Fallback

    hist_movement = get_historical_earnings_movement(symbol)
    if hist_movement is None:
        hist_movement = 5.0  # Fallback

    eps_impact = abs(eps_est) * iv * 10
    revenue_impact = (revenue_est / 1_000_000_000) * 0.05
    time_factor = 1.0 if time == "bmo" else 1.2 if time == "amc" else 0.8

    movement = (hist_movement * 0.5 + iv * 100 * 0.3 + eps_impact * 0.15 + revenue_impact * 0.05) * time_factor
    return round(max(1.0, min(50.0, movement)), 2)














    # Fallback to FMP API
    url_fmp = f"{FMP_BASE_URL}/quote/{ticker}"
    params_fmp = {"apikey": FMP_API_KEY}
    try:
        response = session_fmp.get(url_fmp, params=params_fmp, headers=HEADERS_FMP, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            price = float(data[0].get("price", 0.0))
            if price > 0:
                logger.info(f"Fetched current price for {ticker} from FMP: ${price:.2f}")
                return price
    except Exception as e:
        logger.error(f"FMP failed to fetch price for {ticker}: {str(e)}")

    logger.error(f"Unable to fetch current price for {ticker} from any API")
    return 0.0


# Definiciones de funciones necesarias antes de main()
# Definiciones de funciones necesarias antes de main()
# Definiciones de funciones necesarias antes de main()
def fetch_api_data(url: str, params: Dict, headers: Dict, source: str) -> Optional[Dict]:
    """
    Realiza una solicitud GET a una API con manejo de reintentos y logging.
    """
    session = session_fmp if "FMP" in source else session_tradier
    try:
        response = session.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        logger.debug(f"{source} fetch success: {len(response.text)} bytes")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"{source} error: {e}")
        return None

@st.cache_data(ttl=60)
def get_current_price(ticker: str) -> float:
    """
    Obtiene el precio actual de un ticker usando la API de Tradier con fallback a FMP.
    """
    url_tradier = f"{TRADIER_BASE_URL}/markets/quotes"
    params_tradier = {"symbols": ticker}
    try:
        response = session_tradier.get(url_tradier, params=params_tradier, headers=HEADERS_TRADIER, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and "quotes" in data and "quote" in data["quotes"]:
            quote = data["quotes"]["quote"]
            if isinstance(quote, list):
                quote = quote[0]
            price = float(quote.get("last", 0.0))
            if price > 0:
                logger.info(f"Fetched current price for {ticker} from Tradier: ${price:.2f}")
                return price
    except Exception as e:
        logger.warning(f"Failed to fetch price for {ticker}: {str(e)}")

    url_fmp = f"{FMP_BASE_URL}/quote/{ticker}"
    params_fmp = {"apikey": FMP_API_KEY}
    try:
        response = session_fmp.get(url_fmp, params=params_fmp, headers=HEADERS_FMP, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            price = float(data[0].get("price", 0.0))
            if price > 0:
                logger.info(f"Fetched current price for {ticker} from FMP: ${price:.2f}")
                return price
    except Exception as e:
        logger.error(f"FMP failed to fetch price for {ticker}: {str(e)}")

    logger.error(f"Unable to fetch current price for {ticker} from any API")
    return 0.0

@st.cache_data(ttl=86400)
def get_expiration_dates(ticker: str) -> List[str]:
    """
    Obtiene las fechas de vencimiento de opciones para un ticker dado usando la API de Tradier.
    """
    url = f"{TRADIER_BASE_URL}/markets/options/expirations"
    params = {"symbol": ticker}
    try:
        response = session_tradier.get(url, params=params, headers=HEADERS_TRADIER, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and "expirations" in data and "date" in data["expirations"]:
            expiration_dates = data["expirations"]["date"]
            logger.info(f"Fetched {len(expiration_dates)} expiration dates for {ticker}")
            return expiration_dates
        logger.warning(f"No expiration dates found for {ticker}")
        return []
    except Exception as e:
        logger.error(f"Error fetching expiration dates for {ticker}: {str(e)}")
        return []

@st.cache_data(ttl=60)
def get_current_prices(tickers: List[str]) -> Dict[str, float]:
    """
    Obtiene precios actuales para una lista de tickers usando la API de Tradier con fallback a FMP.
    """
    prices_dict = {ticker: 0.0 for ticker in tickers}
    
    tickers_str = ",".join(tickers)
    url_tradier = f"{TRADIER_BASE_URL}/markets/quotes"
    params_tradier = {"symbols": tickers_str}
    try:
        response = session_tradier.get(url_tradier, params=params_tradier, headers=HEADERS_TRADIER, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and "quotes" in data and "quote" in data["quotes"]:
            quotes = data["quotes"]["quote"]
            if isinstance(quotes, dict):
                quotes = [quotes]
            for quote in quotes:
                ticker = quote.get("symbol", "")
                price = float(quote.get("last", 0.0))
                if ticker in prices_dict and price > 0:
                    prices_dict[ticker] = price
            fetched = [t for t, p in prices_dict.items() if p > 0]
            logger.info(f"Fetched prices for {len(fetched)}/{len(tickers)} tickers from Tradier: {fetched}")
    except Exception as e:
        logger.warning(f"Tradier failed to fetch prices for batch: {str(e)}")

    missing_tickers = [t for t, p in prices_dict.items() if p == 0.0]
    if missing_tickers:
        url_fmp = f"{FMP_BASE_URL}/quote/{','.join(missing_tickers)}"
        params_fmp = {"apikey": FMP_API_KEY}
        try:
            response = session_fmp.get(url_fmp, params=params_fmp, headers=HEADERS_FMP, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list):
                for item in data:
                    ticker = item.get("symbol", "")
                    price = float(item.get("price", 0.0))
                    if ticker in prices_dict and price > 0:
                        prices_dict[ticker] = price
                fetched = [t for t, p in prices_dict.items() if p > 0 and t in missing_tickers]
                logger.info(f"Fetched prices for {len(fetched)}/{len(missing_tickers)} missing tickers from FMP: {fetched}")
        except Exception as e:
            logger.error(f"FMP failed to fetch prices for batch: {str(e)}")

    failed = [t for t, p in prices_dict.items() if p == 0.0]
    if failed:
        logger.error(f"Unable to fetch prices for {len(failed)} tickers: {failed}")

    return prices_dict

@st.cache_data(ttl=3600)
def get_metaverse_stocks() -> List[str]:
    """
    Obtiene una lista de los 50 stocks más activos desde FMP o usa una lista de respaldo.
    """
    url = "https://financialmodelingprep.com/api/v3/stock_market/actives"
    params = {"apikey": FMP_API_KEY}
    data = fetch_api_data(url, params, HEADERS_FMP, "FMP Actives")
    if data and isinstance(data, list):
        return [stock["symbol"] for stock in data[:50]]
    logger.warning("Failed to retrieve stocks from FMP API. Using fallback list.")
    return ["NVDA", "TSLA", "AAPL", "AMD", "PLTR", "META", "RBLX", "U", "COIN", "HOOD"]












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
    combined_tickers = set()  # Usamos un set para evitar duplicados

    # 1. Obtener lista de FMP
    try:
        response = requests.get(
            f"{FMP_BASE_URL}/stock-screener",
            params={
                "apikey": FMP_API_KEY,
                "marketCapMoreThan": 1_000_000_000,  # Capitalización > $1B
                "volumeMoreThan": 500_000,           # Volumen > 500k
                "priceMoreThan": 5,                  # Precio > $5
                "exchange": "NASDAQ,NYSE"            # Solo NASDAQ y NYSE
            }
        )
        response.raise_for_status()
        data = response.json()
        fmp_tickers = [stock["symbol"] for stock in data if stock.get("isActivelyTrading", True)]
        combined_tickers.update(fmp_tickers[:200])  # Limitamos a 200 por velocidad
        logger.info(f"FMP returned {len(fmp_tickers)} tickers")
    except Exception as e:
        logger.error(f"FMP stock list failed: {str(e)}")

    # 2. Obtener lista de Tradier (usamos endpoint de quotes con múltiples símbolos)
    try:
        # Tradier no tiene un endpoint directo de "screener", así que usamos una lista inicial de índices o ETFs populares
        initial_tickers = "SPY,QQQ,DIA,IWM,TSLA,AAPL,MSFT,NVDA,GOOGL,AMZN,META"  # Base inicial
        url_tradier = f"{TRADIER_BASE_URL}/markets/quotes"
        params_tradier = {"symbols": initial_tickers}
        data_tradier = fetch_api_data(url_tradier, params_tradier, HEADERS_TRADIER, "Tradier")
        if data_tradier and "quotes" in data_tradier and "quote" in data_tradier["quotes"]:
            quotes = data_tradier["quotes"]["quote"]
            if isinstance(quotes, dict):
                quotes = [quotes]
            tradier_tickers = [
                quote["symbol"] for quote in quotes
                if quote.get("last", 0) > 5 and quote.get("volume", 0) > 500_000
            ]
            combined_tickers.update(tradier_tickers)
            logger.info(f"Tradier returned {len(tradier_tickers)} tickers")
    except Exception as e:
        logger.error(f"Tradier stock list failed: {str(e)}")

    # Convertimos a lista y limitamos el resultado
    final_list = list(combined_tickers)
    logger.info(f"Combined unique tickers: {len(final_list)}")
    return final_list[:200]  # Máximo 200 para mantener rendimiento

# --- Funciones de Análisis ---
def analyze_contracts(ticker, expiration, current_price):
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {"symbol": ticker, "expiration": expiration, "greeks": True}
    try:
        response = requests.get(url, headers=HEADERS_TRADIER, params=params, timeout=10)  # Timeout de 10 segundos
        if response.status_code != 200:
            st.error(f"Error retrieving option contracts: {response.status_code}")
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
    except requests.exceptions.ReadTimeout:
        st.error(f"Timeout retrieving option contracts for {ticker}. Tradier API did not respond.")
        logger.error(f"ReadTimeout in analyze_contracts for {ticker}, expiration {expiration}")
        return pd.DataFrame()
    except requests.RequestException as e:
        st.error(f"Error retrieving option contracts for {ticker}: {str(e)}")
        logger.error(f"RequestException in analyze_contracts: {str(e)}")
        return pd.DataFrame()

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

# Versión original para opciones (usada en Tab 1)
# Versión original para opciones (usada en Tab 1)
def calculate_max_pain(df):
    """Calcula el Max Pain para opciones."""
    if df.empty or 'strike' not in df.columns:
        logger.error("DataFrame vacío o sin columna 'strike' en calculate_max_pain")
        return None, pd.DataFrame(columns=['strike', 'total_loss'])
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
        logger.warning("No se generaron datos de Max Pain")
        return None, max_pain_df
    max_pain_strike = max_pain_df.loc[max_pain_df['total_loss'].idxmin()]
    return max_pain_strike, max_pain_df.sort_values(by='total_loss', ascending=True)

def calculate_support_resistance_mid(max_pain_table, current_price):
    """Calcula niveles de soporte y resistencia basados en Max Pain."""
    if max_pain_table.empty or 'strike' not in max_pain_table.columns:
        return current_price, current_price, current_price
    puts = max_pain_table[max_pain_table['strike'] <= current_price]
    calls = max_pain_table[max_pain_table['strike'] > current_price]
    support_level = puts.loc[puts['total_loss'].idxmin()]['strike'] if not puts.empty else current_price
    resistance_level = calls.loc[calls['total_loss'].idxmin()]['strike'] if not calls.empty else current_price
    mid_level = (support_level + resistance_level) / 2
    return support_level, resistance_level, mid_level

def plot_max_pain_histogram_with_levels(max_pain_table, current_price):
    """Crea un histograma de Max Pain con niveles."""
    if max_pain_table.empty:
        fig = go.Figure()
        fig.update_layout(title="Max Pain Histogram (No Data)", template="plotly_white")
        return fig
    
    support_level, resistance_level, mid_level = calculate_support_resistance_mid(max_pain_table, current_price)
    max_pain_table['loss_category'] = max_pain_table['total_loss'].apply(
        lambda x: 'High Loss' if x > max_pain_table['total_loss'].quantile(0.75) else ('Low Loss' if x < max_pain_table['total_loss'].quantile(0.25) else 'Neutral')
    )
    color_map = {'High Loss': '#FF5733', 'Low Loss': '#28A745', 'Neutral': 'rgba(128,128,128,0.3)'}
    fig = px.bar(max_pain_table, x='strike', y='total_loss', title="Max Pain Histogram with Levels",
                 labels={'total_loss': 'Total Loss', 'strike': 'Strike Price'}, color='loss_category', color_discrete_map=color_map)
    fig.update_layout(xaxis_title="Strike Price", yaxis_title="Total Loss", template="plotly_white", font=dict(size=14, family="Open Sans"),
                      title=dict(text="📊 Analysis loss Options", font=dict(size=18), x=0.5), hovermode="x",
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

    # Crear la figura
    fig = go.Figure()

    # Añadir barras para CALLs y PUTs con ancho fijo
    fig.add_trace(go.Bar(
        x=strikes,
        y=gamma_calls,
        name="Gummy CALL",
        marker=dict(color=call_colors),
        width=0.4,
        hovertemplate="Gummy CALL: %{y:.2f}",  # Sin Current Price
    ))
    fig.add_trace(go.Bar(
        x=strikes,
        y=gamma_puts,
        name="Gummy PUT",
        marker=dict(color=put_colors),
        width=0.4,
        hovertemplate="Gummy PUT: %{y:.2f}",  # Sin Current Price
    ))

    # Línea vertical para Current Price
    y_min = min(gamma_calls + gamma_puts) * 1.1
    y_max = max(gamma_calls + gamma_puts) * 1.1
    fig.add_trace(go.Scatter(
        x=[current_price, current_price],
        y=[y_min, y_max],
        mode="lines",
        line=dict(width=1, dash="dot", color="#39FF14"),
        name="Current Price",
        hovertemplate="",  # Tooltip vacío para evitar redundancia
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0)",  # Fondo completamente transparente
            bordercolor="rgba(0,0,0,0)",  # Borde completamente transparente
            font=dict(color="#39FF14", size=12)  # Letras verdes "en el aire"
        )
    ))

    # Añadir label fijo profesional para Current Price
    fig.add_annotation(
        x=current_price,
        y=y_max * 0.95,  # Posición cerca del tope del gráfico
        text=f"Price: ${current_price:.2f}",
        showarrow=False,
        font=dict(color="#39FF14", size=10),  # Verde, pequeño y profesional
        bgcolor="rgba(0,0,0,0.5)",  # Fondo semitransparente oscuro
        bordercolor="#39FF14",  # Borde verde fino
        borderwidth=1,
        borderpad=4  # Espacio interno para un look limpio
    )

    # Configuración de los tooltips y layout
    fig.update_traces(
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.1)",  # Fondo muy transparente para las barras
            bordercolor="rgba(255,255,255,0.3)",  # Borde casi invisible para las barras
            font=dict(color="white", size=12)  # Texto blanco para las barras
        )
    )
    fig.update_layout(
        title="GUMMY EXPOSURE",
        xaxis_title="Strike",
        yaxis_title="Gummy Exposure",
        template="plotly_dark",
        hovermode="x",
        xaxis=dict(
            tickmode="array",
            tickvals=strikes,
            ticktext=[f"{s:.2f}" for s in strikes],
            rangeslider=dict(visible=False),
            showgrid=False
        ),
        yaxis=dict(showgrid=False),
        bargap=0.2,
        barmode="relative",
    )

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
    fig.update_layout(xaxis_title="Strike Price", yaxis_title="Gummy Bubbles® (%)", legend_title="Option Type", template="plotly_white", title_x=0.5)

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
        st.warning(f"Error fetching Data News: {e}")
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

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    prices = np.array(prices)
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    return 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100

def calculate_sma(prices: List[float], period: int = 20) -> Optional[float]:
    prices = np.array(prices)
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def scan_stock_batch(tickers: List[str], scan_type: str, breakout_period=10, volume_threshold=2.0) -> List[Dict]:
    prices_dict = get_current_prices(tickers)
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(get_historical_prices_combined, ticker, limit=breakout_period+1): ticker for ticker in tickers}
        for future in futures:
            ticker = futures[future]
            try:
                prices, volumes = future.result()
                if len(prices) <= breakout_period or not volumes:
                    continue
                current_price = prices_dict.get(ticker, 0.0)
                if current_price == 0.0:
                    continue
                prices = np.array(prices)
                volumes = np.array(volumes)
                rsi = calculate_rsi(prices)
                sma = calculate_sma(prices)
                avg_volume = np.mean(volumes)
                current_volume = volumes[-1]
                recent_high = np.max(prices[-breakout_period:])
                recent_low = np.min(prices[-breakout_period:])
                last_price = prices[-1]
                near_support = abs(last_price - recent_low) / recent_low <= 0.05
                near_resistance = abs(last_price - recent_high) / recent_high <= 0.05
                breakout_type = "Up" if last_price > recent_high else "Down" if last_price < recent_low else None
                possible_change = (recent_low - last_price) / last_price * 100 if near_support else (recent_high - last_price) / last_price * 100 if near_resistance else None

                if scan_type == "Bullish (Upward Momentum)" and sma and last_price > sma and rsi and rsi < 70:
                    results.append({"Symbol": ticker, "Last Price": last_price, "SMA": round(sma, 2), "RSI": round(rsi, 2), "Volume": current_volume, "Breakout Type": breakout_type, "Possible Change (%)": round(possible_change, 2) if possible_change else None})
                elif scan_type == "Bearish (Downward Momentum)" and sma and last_price < sma and rsi and rsi > 30:
                    results.append({"Symbol": ticker, "Last Price": last_price, "SMA": round(sma, 2), "RSI": round(rsi, 2), "Volume": current_volume, "Breakout Type": breakout_type, "Possible Change (%)": round(possible_change, 2) if possible_change else None})
                elif scan_type == "Breakouts" and breakout_type:
                    results.append({"Symbol": ticker, "Breakout Type": breakout_type, "Last Price": last_price, "Recent High": recent_high, "Recent Low": recent_low, "Volume": current_volume, "Possible Change (%)": round(possible_change, 2) if possible_change else None})
                elif scan_type == "Unusual Volume" and current_volume > volume_threshold * avg_volume:
                    results.append({"Symbol": ticker, "Volume": current_volume, "Avg Volume": avg_volume, "Last Price": last_price})
            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
    return results

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
                st.warning(f"No Data: {endpoint}")
                return None
            return data
        st.error(f"Error al obtener datos: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        st.error(f"Error  HTTP: {str(e)}")
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



    with tab7:
        st.subheader("Elliott Pulse")
        ticker = st.text_input("Ticker Symbol (e.g., SPY)", "SPY", key="elliott_ticker").upper()
        expiration_dates = get_expiration_dates(ticker)
        if not expiration_dates:
            st.error(f"No expiration dates found for '{ticker}'. Try a valid ticker (e.g., SPY).")
            return
        selected_expiration = st.selectbox("Expiration Date", expiration_dates, key="elliott_exp_date")
        volume_threshold = st.slider("Min Open Interest (millions)", 0.1, 2.0, 0.5, step=0.1, key="elliott_vol") * 1_000_000

        with st.spinner(f"Fetching data for {ticker}..."):
            current_price = get_current_price(ticker)
            if current_price == 0.0:
                st.error(f"Unable to fetch current price for '{ticker}'.")
                return
            options_data = get_options_data(ticker, selected_expiration)
            if not options_data:
                st.error("No options data available.")
                return

            # Procesar datos para gamma y volumen
            strikes_data = {}
            for opt in options_data:
                strike = float(opt.get("strike", 0))
                opt_type = opt.get("option_type", "").upper()
                oi = int(opt.get("open_interest", 0))
                greeks = opt.get("greeks", {})
                gamma = float(greeks.get("gamma", 0)) if isinstance(greeks, dict) else 0
                intrinsic = max(current_price - strike, 0) if opt_type == "CALL" else max(strike - current_price, 0)
                if strike not in strikes_data:
                    strikes_data[strike] = {"CALL": {"OI": 0, "Gamma": 0, "Intrinsic": 0}, "PUT": {"OI": 0, "Gamma": 0, "Intrinsic": 0}}
                strikes_data[strike][opt_type]["OI"] += oi
                strikes_data[strike][opt_type]["Gamma"] += gamma * oi  # Gamma ponderado por OI
                strikes_data[strike][opt_type]["Intrinsic"] = intrinsic

            # Filtrar strikes con OI >= threshold y calcular gamma neto
            strikes = sorted(strikes_data.keys())
            call_gamma = []
            put_gamma = []
            net_gamma = []
            intrinsic_values = []
            for strike in strikes:
                call_oi = strikes_data[strike]["CALL"]["OI"]
                put_oi = strikes_data[strike]["PUT"]["OI"]
                if call_oi >= volume_threshold or put_oi >= volume_threshold:
                    cg = strikes_data[strike]["CALL"]["Gamma"]
                    pg = strikes_data[strike]["PUT"]["Gamma"]
                    call_gamma.append(cg)
                    put_gamma.append(-pg)
                    net_gamma.append(cg - pg)
                    intrinsic_values.append(max(strikes_data[strike]["CALL"]["Intrinsic"], strikes_data[strike]["PUT"]["Intrinsic"]))
                else:
                    call_gamma.append(0)
                    put_gamma.append(0)
                    net_gamma.append(0)
                    intrinsic_values.append(0)

            # Encontrar el strike con mayor gamma neto absoluto más cercano al precio actual
            nearest_strike_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - current_price) if abs(net_gamma[i]) > 0 else float('inf'))
            if nearest_strike_idx == float('inf'):
                st.warning("No significant gamma found above volume threshold.")
                return
            target_strike = strikes[nearest_strike_idx]
            target_gamma = net_gamma[nearest_strike_idx]
            predicted_move = "Up" if target_gamma > 0 else "Down"

            # Crear gráfica
            fig = go.Figure()
            fig.add_trace(go.Bar(x=strikes, y=call_gamma, name="CALL Gamma", marker_color="green", width=0.4))
            fig.add_trace(go.Bar(x=strikes, y=put_gamma, name="PUT Gamma", marker_color="red", width=0.4))
            fig.add_trace(go.Scatter(x=[current_price, current_price], y=[min(put_gamma) * 1.1, max(call_gamma) * 1.1], 
                                    mode="lines", line=dict(color="#39FF14", dash="dash"), name="Current Price"))
            fig.add_trace(go.Scatter(x=[target_strike], y=[target_gamma], mode="markers+text", marker=dict(size=15, color="yellow"),
                                    text=[f"Target: ${target_strike:.2f}"], textposition="top center", name="Predicted Move"))

            fig.update_layout(
                title=f"Elliott Pulse {ticker} (Exp: {selected_expiration})",
                xaxis_title="Strike Price",
                yaxis_title="Gummy Exposure",
                barmode="relative",
                template="plotly_dark",
                annotations=[dict(x=target_strike, y=max(call_gamma) * 0.9, text=f"Next Move: {predicted_move}", showarrow=True, arrowhead=2, 
                                font=dict(color="yellow", size=12))]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Predicted Next Move: {predicted_move} towards ${target_strike:.2f} (Intrinsic Value: ${intrinsic_values[nearest_strike_idx]:.2f})")


# --- Nust.cache_data(ttl=CACHE_TTL)
# --- Nuevas funciones para cripto (necesarias para Tab 8) ---
# --- Nuevas funciones para cripto (necesarias para Tab 8) ---
# --- Nuevas funciones para cripto (necesarias para Tab 8) ---
# Línea ~500: Funciones de soporte para el Tab 8
# Línea ~500: Funciones de soporte para el Tab 8
# Línea ~500: Funciones de soporte para el Tab 8


# Línea ~500: Funciones de soporte
# Línea ~500: Funciones de soporte
# Versión original para opciones (usada en Tab 1)
# Línea ~500: Funciones de soporte
# Versión original para opciones (usada en Tab 1)

# Funciones para el Tab 8
# Línea ~500: Funciones de soporte

# Versión original para opciones (usada en Tab 1)
def calculate_max_pain(df):
    """Calcula el Max Pain para opciones."""
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



# Funciones para el Tab 8
# Funciones para el Tab 8
# Funciones para el Tab 8
# Funciones para el Tab 8
def kraken_pair_to_api_format(ticker: str) -> str:
    """Convierte un ticker (e.g., BTC) al formato de Kraken (e.g., XXBTZUSD)."""
    base = ticker.upper()
    quote = "USD"
    if base == "BTC":
        base = "XBT"
    return f"X{base}Z{quote}"

def fetch_order_book(ticker: str, depth: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Obtiene el libro de órdenes en vivo desde Kraken con máxima profundidad."""
    api_pair = kraken_pair_to_api_format(ticker)
    try:
        response = kraken.query_public("Depth", {"pair": api_pair, "count": depth})
        if "error" in response and response["error"]:
            logger.error(f"Error fetching order book for {ticker}/USD: {response['error']}")
            return pd.DataFrame(), pd.DataFrame(), 0.0
        
        result = response["result"][api_pair]
        bids = pd.DataFrame(result["bids"], columns=["Price", "Volume", "Timestamp"]).astype(float)
        asks = pd.DataFrame(result["asks"], columns=["Price", "Volume", "Timestamp"]).astype(float)
        
        if bids.empty or asks.empty:
            logger.warning(f"Empty order book received for {ticker}/USD: bids={len(bids)}, asks={len(asks)}")
        
        best_bid = bids["Price"].max() if not bids.empty else 0
        best_ask = asks["Price"].min() if not asks.empty else 0
        current_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
        
        logger.info(f"Fetched order book for {ticker}/USD: {len(bids)} bids, {len(asks)} asks")
        return bids, asks, current_price
    except Exception as e:
        logger.error(f"Error fetching order book for {ticker}/USD: {e}")
        return pd.DataFrame(), pd.DataFrame(), 0.0

def fetch_coingecko_data(ticker: str) -> dict:
    """Obtiene datos de mercado desde CoinGecko, incluyendo volatilidad."""
    coin_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "XRP": "ripple",
        "LTC": "litecoin",
        "ADA": "cardano"
    }
    coin_id = coin_map.get(ticker.upper(), ticker.lower())
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            logger.error(f"Error fetching  data for {ticker}: {response.status_code}")
            return {}
        data = response.json()
        market_data = data.get("market_data", {})
        # URL para datos históricos de 24h
        history_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1&interval=hourly"
        history_response = requests.get(history_url, timeout=5)
        history_data = history_response.json() if history_response.status_code == 200 else {"prices": []}
        prices = [price[1] for price in history_data.get("prices", [])]
        volatility = stats.stdev([p / prices[0] * 100 - 100 for p in prices]) * (365 ** 0.5) if len(prices) > 1 else 0  # Volatilidad anualizada
        return {
            "price": market_data.get("current_price", {}).get("usd", 0),
            "change_value": market_data.get("price_change_24h", 0),
            "change_percent": market_data.get("price_change_percentage_24h", 0),
            "volume": market_data.get("total_volume", {}).get("usd", 0),
            "market_cap": market_data.get("market_cap", {}).get("usd", 0),
            "volatility": volatility
        }
    except Exception as e:
        logger.error(f"Error fetching  data for {ticker}: {str(e)}")
        return {}

def calculate_crypto_max_pain(bids: pd.DataFrame, asks: pd.DataFrame) -> float:
    """Calcula el Max Pain basado en el libro de órdenes de criptomonedas."""
    if bids.empty or asks.empty:
        return 0.0
    
    all_prices = sorted(set(bids["Price"].tolist() + asks["Price"].tolist()))
    min_price = min(all_prices)
    max_price = max(all_prices)
    price_range = np.linspace(min_price, max_price, 200)  # Más precisión
    
    max_pain_losses = {}
    for price in price_range:
        bid_loss = bids[bids["Price"] < price]["Volume"].sum() * (price - bids[bids["Price"] < price]["Price"]).sum()
        ask_loss = asks[asks["Price"] > price]["Volume"].sum() * (asks[asks["Price"] > price]["Price"] - price).sum()
        total_loss = bid_loss + ask_loss
        max_pain_losses[price] = total_loss
    
    max_pain_price = min(max_pain_losses, key=max_pain_losses.get, default=0.0)
    return max_pain_price

def calculate_metrics_with_whales(bids: pd.DataFrame, asks: pd.DataFrame, current_price: float, market_volatility: float) -> dict:
    """Calcula métricas avanzadas con órdenes de ballenas y volatilidad de mercado."""
    total_bid_volume = bids["Volume"].sum() if not bids.empty else 0
    total_ask_volume = asks["Volume"].sum() if not asks.empty else 0
    total_volume = total_bid_volume + total_ask_volume
    net_pressure = total_bid_volume - total_ask_volume if total_volume > 0 else 0
    pressure_index = (net_pressure / total_volume * 100) if total_volume > 0 else 0  # Índice de presión
    
    # Órdenes de ballenas
    whale_threshold = max(bids["Volume"].quantile(0.95) if not bids.empty else 0, 
                          asks["Volume"].quantile(0.95) if not asks.empty else 0, 
                          50.0)  # Top 5% o 50 unidades
    whale_bids = bids[bids["Volume"] >= whale_threshold] if not bids.empty else pd.DataFrame()
    whale_asks = asks[asks["Volume"] >= whale_threshold] if not asks.empty else pd.DataFrame()
    
    whale_bid_volume = whale_bids["Volume"].sum() if not whale_bids.empty else 0
    whale_ask_volume = whale_asks["Volume"].sum() if not whale_asks.empty else 0
    whale_net_pressure = whale_bid_volume - whale_ask_volume
    whale_pressure_weight = (whale_bid_volume + whale_ask_volume) / total_volume if total_volume > 0 else 0
    
    whale_bid_price = (whale_bids["Price"] * whale_bids["Volume"]).sum() / whale_bid_volume if whale_bid_volume > 0 else current_price
    whale_ask_price = (whale_asks["Price"] * whale_asks["Volume"]).sum() / whale_ask_volume if whale_ask_volume > 0 else current_price
    
    # Support y Resistance basados en acumulaciones de volumen
    bids["CumVolume"] = bids["Volume"].cumsum()
    asks["CumVolume"] = asks["Volume"].cumsum()
    support = bids[bids["CumVolume"] >= total_bid_volume * 0.25]["Price"].min() if not bids.empty else current_price  # 25% del volumen bid
    resistance = asks[asks["CumVolume"] >= total_ask_volume * 0.25]["Price"].max() if not asks.empty else current_price  # 25% del volumen ask
    
    # Whale Accumulation Zones (clústeres)
    whale_zones = []
    if not whale_bids.empty:
        whale_zones.extend(whale_bids["Price"].tolist())
    if not whale_asks.empty:
        whale_zones.extend(whale_asks["Price"].tolist())
    whale_zones = sorted(set(whale_zones))[:6]  # Limitar a 6 zonas
    
    # Fórmula personalizada para target (mi toque especial)
    max_pain_price = calculate_crypto_max_pain(bids, asks)
    if max_pain_price != 0.0 and current_price != 0.0:
        distance_to_max_pain = max_pain_price - current_price
        whale_influence = (whale_bid_price * whale_bid_volume - whale_ask_price * whale_ask_volume) / (whale_bid_volume + whale_ask_volume + 1) if (whale_bid_volume + whale_ask_volume) > 0 else 0
        whale_factor = whale_pressure_weight * whale_influence * 3  # Más peso a ballenas
        volatility_factor = market_volatility / 100  # Volatilidad de CoinGecko como amplificador
        possible_move = (distance_to_max_pain * (pressure_index / 100) + whale_factor) * (1 + volatility_factor)
        target_price = current_price + possible_move
        direction = "BUY" if current_price < target_price else "SELL" if current_price > target_price else "HOLD"
        
        # Trader's Edge Score (mi sorpresa)
        whale_momentum = whale_net_pressure / (whale_bid_volume + whale_ask_volume + 1) * 100 if (whale_bid_volume + whale_ask_volume) > 0 else 0
        edge_score = (pressure_index * 0.4 + whale_momentum * 0.4 + volatility_factor * 20)  # Puntuación de 0 a 100
    else:
        target_price = current_price
        direction = "HOLD"
        edge_score = 0
    
    return {
        "net_pressure": net_pressure,
        "volatility": market_volatility,  # De CoinGecko
        "support": support,
        "resistance": resistance,
        "whale_zones": whale_zones,
        "target_price": target_price,
        "direction": direction,
        "trend": "Bullish" if net_pressure > 0 else "Bearish" if net_pressure < 0 else "Neutral",
        "whale_bids": whale_bids,
        "whale_asks": whale_asks,
        "edge_score": edge_score
    }

def plot_order_book_bubbles_with_max_pain(bids: pd.DataFrame, asks: pd.DataFrame, current_price: float, ticker: str, market_volatility: float) -> Tuple[go.Figure, dict]:
    """Crea un gráfico de burbujas con Max Pain, órdenes de ballenas y niveles clave."""
    fig = go.Figure()
    
    # Órdenes regulares
    if not bids.empty:
        fig.add_trace(go.Scatter(
            x=bids["Price"],
            y=[0] * len(bids),
            mode="markers",
            name="Bids",
            marker=dict(
                size=bids["Volume"] * 20 / bids["Volume"].max(),
                color="#32CD32",
                opacity=0.7,
                line=dict(width=0.5, color="white")
            ),
            customdata=bids[["Price", "Volume"]],
            hovertemplate="<b>Price:</b> $%{customdata[0]:.2f}<br><b>Volume:</b> %{customdata[1]:.2f}"
        ))
    
    if not asks.empty:
        fig.add_trace(go.Scatter(
            x=asks["Price"],
            y=[0] * len(asks),
            mode="markers",
            name="Asks",
            marker=dict(
                size=asks["Volume"] * 20 / asks["Volume"].max(),
                color="#FF4500",
                opacity=0.7,
                line=dict(width=0.5, color="white")
            ),
            customdata=asks[["Price", "Volume"]],
            hovertemplate="<b>Price:</b> $%{customdata[0]:.2f}<br><b>Volume:</b> %{customdata[1]:.2f}"
        ))
    
    metrics = calculate_metrics_with_whales(bids, asks, current_price, market_volatility)
    
    # Resaltar órdenes de ballenas
    if not metrics["whale_bids"].empty:
        fig.add_trace(go.Scatter(
            x=metrics["whale_bids"]["Price"],
            y=[0] * len(metrics["whale_bids"]),
            mode="markers",
            name="Whale Bids",
            marker=dict(
                size=metrics["whale_bids"]["Volume"] * 20 / bids["Volume"].max(),
                color="#00FF00",
                opacity=0.9,
                line=dict(width=2, color="white")
            ),
            customdata=metrics["whale_bids"][["Price", "Volume"]],
            hovertemplate="<b>Whale Bid Price:</b> $%{customdata[0]:.2f}<br><b>Volume:</b> %{customdata[1]:.2f}"
        ))
    
    if not metrics["whale_asks"].empty:
        fig.add_trace(go.Scatter(
            x=metrics["whale_asks"]["Price"],
            y=[0] * len(metrics["whale_asks"]),
            mode="markers",
            name="Whale Asks",
            marker=dict(
                size=metrics["whale_asks"]["Volume"] * 20 / asks["Volume"].max(),
                color="#FF0000",
                opacity=0.9,
                line=dict(width=2, color="white")
            ),
            customdata=metrics["whale_asks"][["Price", "Volume"]],
            hovertemplate="<b>Whale Ask Price:</b> $%{customdata[0]:.2f}<br><b>Volume:</b> %{customdata[1]:.2f}"
        ))
    
    # Líneas de precio actual y target
    if current_price > 0:
        fig.add_vline(
            x=current_price,
            line=dict(color="#FFD700", width=1, dash="dash"),
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#FFD700", size=10)
        )
    
    if metrics["target_price"] != current_price:
        fig.add_vline(
            x=metrics["target_price"],
            line=dict(color="#39FF14", width=1, dash="dot"),
            annotation_text=f"Target: ${metrics['target_price']:.2f} ({metrics['direction']})",
            annotation_position="top right",
            annotation_font=dict(color="#39FF14", size=10)
        )
    
    # Líneas de soporte y resistencia
    fig.add_vline(
        x=metrics["support"],
        line=dict(color="#1E90FF", width=1, dash="dot"),
        annotation_text=f"Support: ${metrics['support']:.2f}",
        annotation_position="bottom left",
        annotation_font=dict(color="#1E90FF", size=8)
    )
    fig.add_vline(
        x=metrics["resistance"],
        line=dict(color="#FF4500", width=1, dash="dot"),
        annotation_text=f"Resistance: ${metrics['resistance']:.2f}",
        annotation_position="bottom right",
        annotation_font=dict(color="#FF4500", size=8)
    )
    
    fig.update_layout(
        title=f"FlowS {ticker}/USD | Strategy",
        xaxis_title="Price (USD)",
        yaxis_title="",
        template="plotly_dark",
        plot_bgcolor="#1E1E1E",
        paper_bgcolor="#1E1E1E",
        font=dict(color="#FFFFFF", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        showlegend=True,
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    
    return fig, metrics























def calculate_volume_power_flow(historical_data, current_price, bin_size=100):
    """Calcular flujo de volumen por precio con Power Index y datos para velas de ballenas."""
    df = pd.DataFrame(historical_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Calcular buy/sell volume
    df["price_change"] = df["close"].diff()
    df["buy_volume"] = df.apply(lambda row: row["volume"] if row["price_change"] > 0 else 0, axis=1)
    df["sell_volume"] = df.apply(lambda row: row["volume"] if row["price_change"] < 0 else 0, axis=1)
    df["net_volume"] = df["buy_volume"] - df["sell_volume"]
    
    # Bins por precio
    min_price = df["close"].min()
    max_price = df["close"].max()
    price_bins = np.arange(min_price - bin_size, max_price + bin_size, bin_size)
    df["price_bin"] = pd.cut(df["close"], bins=price_bins, labels=price_bins[:-1])
    
    flow_data = df.groupby("price_bin").agg({
        "buy_volume": "sum",
        "sell_volume": "sum",
        "net_volume": "sum",
        "close": ["min", "max"]  # Para velas de ballenas
    }).reset_index()
    flow_data.columns = ["price_bin", "buy_volume", "sell_volume", "net_volume", "price_min", "price_max"]
    flow_data["price_bin"] = flow_data["price_bin"].astype(float)
    
    # Power Index
    flow_data["power_index"] = flow_data["net_volume"] / (flow_data["buy_volume"] + flow_data["sell_volume"]).replace(0, 1) * 100
    
    # Soporte y resistencia
    support = flow_data[flow_data["price_bin"] < current_price].nlargest(1, "buy_volume")["price_bin"].iloc[0] if not flow_data[flow_data["price_bin"] < current_price].empty else current_price
    resistance = flow_data[flow_data["price_bin"] > current_price].nlargest(1, "sell_volume")["price_bin"].iloc[0] if not flow_data[flow_data["price_bin"] > current_price].empty else current_price
    
    # Zonas de acumulación (ballenas)
    accumulation_zones = flow_data.nlargest(3, "buy_volume")[["price_bin", "buy_volume", "price_min", "price_max"]]
    
    return flow_data, support, resistance, accumulation_zones

def plot_volume_power_flow(flow_data, current_price, support, resistance, accumulation_zones):
    """Gráfica de Volume Power Flow con velas de ballenas en zonas de acumulación."""
    fig = go.Figure()
    
    # Buy Volume
    fig.add_trace(go.Bar(
        x=flow_data["price_bin"],
        y=flow_data["buy_volume"],
        name="Buy Volume",
        marker_color="#32CD32",
        width=flow_data["price_bin"].diff().mean() * 0.8,
        customdata=flow_data[["buy_volume", "power_index"]],
        hovertemplate="Price: $%{x:.2f}<br>Buy Volume: %{customdata[0]:,.0f}<br>Power Index: %{customdata[1]:.2f}"
    ))
    
    # Sell Volume (negativo)
    fig.add_trace(go.Bar(
        x=flow_data["price_bin"],
        y=-flow_data["sell_volume"],
        name="Sell Volume",
        marker_color="#FF4500",
        width=flow_data["price_bin"].diff().mean() * 0.8,
        customdata=flow_data[["sell_volume", "power_index"]],
        hovertemplate="Price: $%{x:.2f}<br>Sell Volume: %{customdata[0]:,.0f}<br>Power Index: %{customdata[1]:.2f}"
    ))
    
    # Velas de ballenas en zonas de acumulación
    whale_hovertext = [
        f"Whale Zone: ${row['price_bin']:.2f}<br>Range: ${row['price_min']:.2f} - ${row['price_max']:.2f}<br>Buy Volume: {row['buy_volume']:,.0f}"
        for _, row in accumulation_zones.iterrows()
    ]
    whale_candles = go.Candlestick(
        x=accumulation_zones["price_bin"],
        open=accumulation_zones["price_min"],
        high=accumulation_zones["price_max"],
        low=accumulation_zones["price_min"],
        close=accumulation_zones["price_max"],
        name="Whale Accumulation",
        increasing_line_color="#FFC107",  # Amarillo mostaza
        decreasing_line_color="#FFC107",
        line=dict(width=3),
        hovertext=whale_hovertext,
        hoverinfo="text"
    )
    fig.add_trace(whale_candles)
    
    # Líneas clave
    y_max = flow_data["buy_volume"].max() * 1.1
    y_min = -flow_data["sell_volume"].max() * 1.1
    
    fig.add_trace(go.Scatter(
        x=[current_price, current_price],
        y=[y_min, y_max],
        mode="lines",
        line=dict(color="#FFFFFF", dash="dash", width=2),
        name="Current Price",
        hovertemplate="Current Price: $%{x:.2f}"
    ))
    
    fig.add_trace(go.Scatter(
        x=[support, support],
        y=[y_min, y_max],
        mode="lines",
        line=dict(color="#1E90FF", dash="dot", width=2),
        name=f"Support (${support:.2f})",
        hovertemplate="Support: $%{x:.2f}"
    ))
    
    fig.add_trace(go.Scatter(
        x=[resistance, resistance],
        y=[y_min, y_max],
        mode="lines",
        line=dict(color="#FFD700", dash="dot", width=2),
        name=f"Resistance (${resistance:.2f})",
        hovertemplate="Resistance: $%{x:.2f}"
    ))
    
    fig.update_layout(
        title="Power Flow",
        xaxis_title="Price Level (USD)",
        yaxis_title="Volume (Buy/Sell)",
        barmode="relative",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=1.1, xanchor="right", x=1.0),
        height=500
    )
    return fig

def calculate_liquidity_pulse(historical_data, current_price):
    """Calcular pulso de liquidez diario con target proyectado."""
    df = pd.DataFrame(historical_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    df["price_change"] = df["close"].diff()
    df["buy_volume"] = df.apply(lambda row: row["volume"] if row["price_change"] > 0 else 0, axis=1)
    df["sell_volume"] = df.apply(lambda row: row["volume"] if row["price_change"] < 0 else 0, axis=1)
    df["net_volume"] = df["buy_volume"] - df["sell_volume"]
    
    net_pressure = df["net_volume"].sum()
    trend = "Bullish" if df["price_change"].iloc[-5:].mean() > 0 else "Bearish"
    volatility = df["close"].pct_change().std() * np.sqrt(365) * 100
    
    valid_df = df.dropna(subset=["price_change", "net_volume"])
    valid_df = valid_df[valid_df["net_volume"] != 0]
    if not valid_df.empty:
        sensitivity = valid_df["price_change"] / (valid_df["net_volume"] / 1_000_000)
        sensitivity_avg = sensitivity.replace([np.inf, -np.inf], np.nan).mean()
    else:
        sensitivity_avg = 0
    
    last_net_volume = df["net_volume"].iloc[-1] / 1_000_000 if df["net_volume"].iloc[-1] != 0 else 0
    price_target = current_price if pd.isna(sensitivity_avg) or sensitivity_avg == 0 else current_price + (last_net_volume * sensitivity_avg)
    
    return df, net_pressure, trend, volatility, price_target

def plot_liquidity_pulse(df, current_price, price_target):
    """Gráfica de Liquidity Pulse con target."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["buy_volume"],
        name="Buy Volume",
        marker_color="#32CD32",
        customdata=df["buy_volume"],
        hovertemplate="Date: %{x}<br>Buy Volume: %{customdata:,.0f}"
    ))
    
    fig.add_trace(go.Bar(
        x=df["date"],
        y=-df["sell_volume"],
        name="Sell Volume",
        marker_color="#FF4500",
        customdata=df["sell_volume"],
        hovertemplate="Date: %{x}<br>Sell Volume: %{customdata:,.0f}"
    ))
    
    y_min = -df["sell_volume"].max() * 1.1 if df["sell_volume"].max() > 0 else -1
    y_max = df["buy_volume"].max() * 1.1 if df["buy_volume"].max() > 0 else 1
    
    avg_volume = df["volume"].mean()
    fig.add_trace(go.Scatter(
        x=[df["date"].iloc[0], df["date"].iloc[-1]],
        y=[avg_volume, avg_volume],
        mode="lines",
        line=dict(color="#FF4500", dash="dash", width=1),
        name=f"Avg Volume ({avg_volume:,.0f})",
        hovertemplate="Avg Volume: %{y:,.0f}"
    ))
    
    fig.add_trace(go.Scatter(
        x=[df["date"].iloc[-1], df["date"].iloc[-1]],
        y=[y_min, y_max],
        mode="lines+text",
        line=dict(color="#00FFFF", dash="dash", width=2),
        text=["", f"Target: ${price_target:,.2f}"],
        textposition="top center",
        name="Projected Target",
        hovertemplate="Projected Target: $%{x:.2f}"
    ))
    
    fig.update_layout(
        title="Liquidity Pulse",
        xaxis_title="Date",
        yaxis_title="Volume (Buy/Sell)",
        barmode="relative",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=1.1, xanchor="right", x=1.0),
        height=400
    )
    return fig

def get_intraday_data(ticker: str, interval="1min", limit=5) -> Tuple[List[float], List[int]]:
    """Obtiene datos intradiarios para IFM."""
    url = f"{TRADIER_BASE_URL}/markets/history"
    params = {"symbol": ticker, "interval": interval, "start": (datetime.now() - timedelta(minutes=limit)).strftime("%Y-%m-%d %H:%M:%S")}
    data = fetch_api_data(url, params, HEADERS_TRADIER, "Tradier Intraday")
    if data and "history" in data and "day" in data["history"]:
        prices = [float(day["close"]) for day in data["history"]["day"]]
        volumes = [int(day["volume"]) for day in data["history"]["day"]]
        return prices, volumes
    return [0.0] * limit, [0] * limit

def get_vix() -> float:
    """Obtiene el VIX actual."""
    url = f"{FMP_BASE_URL}/quote/^VIX"
    params = {"apikey": FMP_API_KEY}
    data = fetch_api_data(url, params, HEADERS_FMP, "VIX")
    return float(data[0]["price"]) if data and isinstance(data, list) and "price" in data[0] else 20.0  # Fallback

def get_news_sentiment(ticker: str) -> float:
    """Calcula el sentimiento de noticias recientes."""
    keywords = [ticker]
    news = fetch_google_news(keywords)
    if not news:
        return 0.5  # Neutral
    sentiment = sum(1 if "up" in article["title"].lower() else -1 if "down" in article["title"].lower() else 0 for article in news)
    return max(0, min(1, 0.5 + sentiment / (len(news) * 2)))  # Escala 0-1

def calculate_probability_cone(current_price: float, iv: float, days: List[int]) -> Dict:
    """Calcula conos de probabilidad para 68% y 95%."""
    cone = {}
    for day in days:
        sigma = iv * current_price * (day / 365) ** 0.5
        cone[day] = {
            "68_lower": current_price - sigma,
            "68_upper": current_price + sigma,
            "95_lower": current_price - 2 * sigma,
            "95_upper": current_price + 2 * sigma
        }
    return cone

# --- Main App (solo Tab 11 actualizado) ---
def interpret_macro_factors(macro_factors: Dict[str, float], market_direction: str, market_magnitude: float) -> List[str]:
    """Interpreta los datos macroeconómicos y predice implicaciones prácticas."""
    implications = []
    
    # Tasa de la FED
    fed_rate = macro_factors["fed_rate"] * 100  # En porcentaje
    if fed_rate > 5.0:
        implications.append(f"Alta tasa de la FED ({fed_rate:.2f}%): Posible presión bajista en sectores cíclicos como Tecnología y Consumo Cíclico por aumento en costos de endeudamiento.")
    elif fed_rate < 2.0:
        implications.append(f"Baja tasa de la FED ({fed_rate:.2f}%): Potencial alza en Real Estate y Utilities por financiamiento barato; el mercado podría beneficiarse de estímulo.")
    else:
        implications.append(f"Tasa de la FED moderada ({fed_rate:.2f}%): Estabilidad relativa, pero atención a sectores sensibles a tasas como Financieros.")

    # PIB
    gdp = macro_factors["gdp"]  # En trillones
    if gdp > 23.0:
        implications.append(f"PIB fuerte ({gdp:.2f}T): Crecimiento económico sólido podría impulsar Industrials y Energy; mercado alcista posible si se mantiene.")
    elif gdp < 20.0:
        implications.append(f"PIB débil ({gdp:.2f}T): Riesgo de recesión, posible bajada en S&P 500 y sectores cíclicos como Consumer Cyclical.")
    else:
        implications.append(f"PIB estable ({gdp:.2f}T): Crecimiento moderado, favorece sectores defensivos como Healthcare y Utilities.")

    # Inflación (CPI)
    cpi = macro_factors["cpi"] * 100  # En porcentaje
    if cpi > 4.0:
        implications.append(f"Alta inflación ({cpi:.2f}%): Presión en bonos (TLT, IEF) por expectativas de tasas más altas; sectores como Energy podrían beneficiarse.")
    elif cpi < 1.0:
        implications.append(f"Baja inflación ({cpi:.2f}%): Posible deflación, riesgo de bajada en S&P 500 y sectores de consumo; bonos podrían subir.")
    else:
        implications.append(f"Inflación controlada ({cpi:.2f}%): Equilibrio favorable para Tecnología y Financieros, sin presión extrema.")

    # Desempleo
    unemployment = macro_factors["unemployment"] * 100  # En porcentaje
    if unemployment > 6.0:
        implications.append(f"Alto desempleo ({unemployment:.2f}%): Posible bajada en Consumer Cyclical y Industrials por menor gasto; mercado bajista probable.")
    elif unemployment < 3.0:
        implications.append(f"Bajo desempleo ({unemployment:.2f}%): Fuerza laboral sólida, potencial alza en S&P 500 y sectores de consumo como XLY.")
    else:
        implications.append(f"Desempleo moderado ({unemployment:.2f}%): Estabilidad laboral, sin impacto extremo en sectores específicos.")

    # Combinación con predicción del mercado
    if market_direction == "Up":
        implications.append(f"Predicción alcista (Magnitud: {market_magnitude:.2f}%): Con estos factores macro, espera subidas en Tecnología y Financieros si la FED no sube tasas abruptamente.")
    elif market_direction == "Down":
        implications.append(f"Predicción bajista (Magnitud: {market_magnitude:.2f}%): Riesgo de caídas en S&P 500 y sectores cíclicos; refúgiate en Utilities o bonos si la inflación o tasas suben.")
    else:
        implications.append(f"Predicción neutral (Magnitud: {market_magnitude:.2f}%): Mercado lateral, busca oportunidades en sectores defensivos como Healthcare o ajusta según noticias macro.")

    return implications

@st.cache_data(ttl=86400)
def get_macro_data(indicator: str) -> float:
    """Obtiene datos macroeconómicos recientes desde FMP."""
    url = f"{FMP_BASE_URL}/economic?name={indicator}"
    params = {"apikey": FMP_API_KEY}
    data = fetch_api_data(url, params, HEADERS_FMP, f"FMP Macro {indicator}")
    if data and isinstance(data, list) and len(data) > 0:
        value = float(data[0].get("value", 0.0))
        if indicator in ["CPI", "CORE_CPI", "PPI", "PCE", "FEDFUNDS", "UNEMPLOYMENT", "CCI", "JOLTS", "ISM_SERVICES"]:  # Porcentajes o índices
            value = value / 100 if indicator in ["CPI", "CORE_CPI", "PPI", "PCE", "FEDFUNDS", "UNEMPLOYMENT"] else value
        elif indicator in ["GDP"]:  # Billones
            value = value / 1_000_000_000_000
        logger.info(f"{indicator}: {value}")
        return value
    logger.warning(f"No {indicator} data available, using fallback")
    return {
        "FEDFUNDS": 0.045, "GDP": 20.0, "CPI": 0.03, "CORE_CPI": 0.03, "PPI": 0.03, "PCE": 0.02,
        "UNEMPLOYMENT": 0.04, "CCI": 100.0, "JOLTS": 7.0, "ISM_SERVICES": 50.0, "TREASURY_10Y": 0.04
    }.get(indicator, 0.0)

def get_macro_factors() -> Dict[str, float]:
    """Obtiene un conjunto ampliado de factores macroeconómicos."""
    return {
        "fed_rate": get_macro_data("FEDFUNDS"),         # Tasa de fondos federales
        "gdp": get_macro_data("GDP"),                   # PIB
        "cpi": get_macro_data("CPI"),                   # Inflación general
        "core_cpi": get_macro_data("CORE_CPI"),         # Inflación subyacente
        "ppi": get_macro_data("PPI"),                   # Índice de precios al productor
        "pce": get_macro_data("PCE"),                   # Gastos de consumo personal
        "unemployment": get_macro_data("UNEMPLOYMENT"), # Tasa de desempleo
        "cci": get_macro_data("CCI"),                   # Confianza del consumidor
        "jolts": get_macro_data("JOLTS"),               # Ofertas de empleo
        "ism_services": get_macro_data("ISM_SERVICES"), # ISM Servicios
        "treasury_10y": get_macro_data("TREASURY_10Y")  # Rendimiento de bonos a 10 años
    }

def calculate_performance(start_price: float, end_price: float) -> float:
    """Calcula el rendimiento porcentual entre dos precios."""
    if start_price and end_price and start_price > 0:
        return ((end_price - start_price) / start_price) * 100
    return 0.0

@st.cache_data(ttl=86400)
def get_fed_rate() -> float:
    """Obtiene la tasa de fondos federales más reciente desde FMP (proxy para FRED)."""
    url = f"{FMP_BASE_URL}/economic?name=FEDFUNDS"
    params = {"apikey": FMP_API_KEY}
    data = fetch_api_data(url, params, HEADERS_FMP, "FMP Fed Rate")
    if data and isinstance(data, list) and len(data) > 0:
        rate = float(data[0].get("value", 0.0)) / 100  # Convertir de porcentaje a decimal
        logger.info(f"Fed Funds Rate: {rate}")
        return rate
    logger.warning("No Fed rate data available, ")
    return 0.045  # Fallback a tasa libre de riesgo


def calculate_momentum(prices: List[float], vol_historical: float) -> float:
    """
    Calculate momentum based on price changes and historical volatility.
    
    Args:
        prices (List[float]): List of historical prices.
        vol_historical (float): Annualized historical volatility.
    
    Returns:
        float: Momentum score.
    """
    if len(prices) < 2:
        return 0.0
    price_change = (prices[-1] - prices[-2]) / prices[-2]
    return price_change / vol_historical if vol_historical > 0 else 0.0




# --- Main App --
# --- Main App ---
# --- Main App ---
# --- Main App ---
def main():
    # Inicialización diferida de la base de datos de contraseñas
    if not os.path.exists(PASSWORDS_DB):
        initialize_passwords_db()

    # Logo y título principal con diseño mejorado
    st.markdown("""
        <style>
        .header-container {
            background: linear-gradient(90deg, #1B263B 0%, #2D2D2D 100%);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .header-title {
            font-size: 36px;
            font-weight: 700;
            color: #FFFFFF;
            text-shadow: 0 0 10px rgba(50, 205, 50, 0.8);
            letter-spacing: 2px;
            text-align: center;
            font-family: 'Roboto', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
            <div class="header-container">
                <div class="header-title">ℙℝ𝕆 𝔼𝕊ℂ𝔸ℕℕ𝔼ℝ®</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        logo_path = "assets/favicon.png"
        if not os.path.exists(logo_path):
            logo_path = "favicon.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=60)
        else:
            st.warning("favicon.png not found. Place it in 'C:/Users/urbin/TradingApp/' or 'assets/'.")

    # Estilos personalizados con tabs ultra compactos y futuristas en dos líneas
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
        }
        .stTextInput, .stSelectbox {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }
        .stSpinner > div > div {
            border-color: #32CD32 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            background: linear-gradient(90deg, #1B263B 0%, #2D2D2D 100%);
            padding: 5px;
            gap: 2px;
            border-radius: 8px;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.7);
            margin-top: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 5px 10px;
            margin: 2px;
            color: #E0E1DD;
            background: #2D2D2D;
            border-radius: 5px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #32CD32;
            color: #1E1E1E;
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(50, 205, 50, 0.5);
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #32CD32;
            color: #1E1E1E;
            font-weight: 700;
            box-shadow: 0 0 15px rgba(50, 205, 50, 0.7);
        }
        </style>
    """, unsafe_allow_html=True)

    # Definición de los tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "Gummy Data Bubbles® |", "Market Scanner |", "News |", "Institutional Holders |",
        "Options Order Flow |", "Analyst Rating Flow |", "Elliott Pulse® |", "Crypto Insights |",
        "Earnings Calendar |", "Psychological Edge |", "Projection |", "Performance Map |"
    ])

    # Tab 1: Gummy Data Bubbles®
    with tab1:
        ticker = st.text_input("Ticker", value="SPY", key="ticker_input").upper()
        
        expiration_dates = get_expiration_dates(ticker)
        if not expiration_dates:
            st.error(f"What were you thinking, '{ticker}'? You're a trader and you mess this up? If you trade like this, you're doomed!")
            return
        
        expiration_date = st.selectbox("Expiration Date", expiration_dates, key="expiration_date")
        
        with st.spinner("Fetching price..."):
            current_price = get_current_price(ticker)
            if current_price == 0.0:
                st.error(f"Unable to fetch current price for '{ticker}'. Check ticker validity, API keys, or internet connection.")
                logger.error(f"Price fetch failed for {ticker}")
                return
        
        st.markdown(f"**Current Price:** ${current_price:.2f}")
        
        with st.spinner(f"Fetching data for {expiration_date}..."):
            options_data = get_options_data(ticker, expiration_date)
            if not options_data:
                st.error("No options data available for this ticker and expiration date.")
                return
            
            # Procesamiento vectorizado corregido
            strikes = np.array([float(opt.get("strike", 0)) for opt in options_data if opt and isinstance(opt, dict)])
            option_types = np.array([opt.get("option_type", "").upper() for opt in options_data if opt and isinstance(opt, dict)])
            ois = np.array([int(opt.get("open_interest", 0)) for opt in options_data if opt and isinstance(opt, dict)])
            gammas = np.array([float(opt.get("greeks", {}).get("gamma", 0)) if isinstance(opt.get("greeks", {}), dict) else 0
                   for opt in options_data if opt and isinstance(opt, dict)])
            
            processed_data = {}
            unique_strikes = np.unique(strikes)
            for strike in unique_strikes:
                mask = strikes == strike
                processed_data[strike] = {
                    "CALL": {
                        "OI": np.sum(ois[mask & (option_types == "CALL")]),
                        "Gamma": np.sum(gammas[mask & (option_types == "CALL")])
                    },
                    "PUT": {
                        "OI": np.sum(ois[mask & (option_types == "PUT")]),
                        "Gamma": np.sum(gammas[mask & (option_types == "PUT")])
                    }
                }
            
            if not processed_data:
                st.error("No valid data to display.")
                return
            
            prices, _ = get_historical_prices_combined(ticker)
            historical_prices = prices
            touched_strikes = detect_touched_strikes(processed_data.keys(), historical_prices)
            max_pain = calculate_max_pain_optimized(options_data)
            df = analyze_contracts(ticker, expiration_date, current_price)
            max_pain_strike, max_pain_df = calculate_max_pain(df)
            
            gamma_fig = gamma_exposure_chart(processed_data, current_price, touched_strikes)
            st.plotly_chart(gamma_fig, use_container_width=True)
            
            gamma_df = pd.DataFrame({
                "Strike": list(processed_data.keys()),
                "CALL_Gamma": [processed_data[s]["CALL"]["Gamma"] for s in processed_data],
                "PUT_Gamma": [processed_data[s]["PUT"]["Gamma"] for s in processed_data],
                "CALL_OI": [processed_data[s]["CALL"]["OI"] for s in processed_data],
                "PUT_OI": [processed_data[s]["PUT"]["OI"] for s in processed_data]
            })
            gamma_csv = gamma_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Gamma Exposure Data",
                data=gamma_csv,
                file_name=f"{ticker}_gamma_exposure_{expiration_date}.csv",
                mime="text/csv",
                key="download_gamma_tab1"
            )
            
            skew_fig, total_calls, total_puts = plot_skew_analysis_with_totals(options_data, current_price)
            st.plotly_chart(skew_fig, use_container_width=True)
            st.write(f"**Total CALLS:** {total_calls} | **Total PUTS:** {total_puts}")
            
            skew_df = pd.DataFrame(options_data)[["strike", "option_type", "open_interest", "volume"]]
            skew_csv = skew_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Skew Analysis Data",
                data=skew_csv,
                file_name=f"{ticker}_skew_analysis_{expiration_date}.csv",
                mime="text/csv",
                key="download_skew_tab1"
            )
            
            st.write(f"Current Price of {ticker}: ${current_price:.2f} (Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            st.write(f"**Max Pain Strike (Optimized):** {max_pain if max_pain else 'N/A'}")
            
            max_pain_fig = plot_max_pain_histogram_with_levels(max_pain_df, current_price)
            st.plotly_chart(max_pain_fig, use_container_width=True)
            
            max_pain_csv = max_pain_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Max Pain Data",
                data=max_pain_csv,
                file_name=f"{ticker}_max_pain_{expiration_date}.csv",
                mime="text/csv",
                key="download_max_pain_tab1"
            )
            
            st.markdown("---")
            st.markdown("*Developed by Ozy | © 2025*")

    # Tab 2: Market Scanner
    with tab2:
        st.subheader("Market Scanner Pro")
        
        # Selección de tipo de escaneo y máximo de resultados
        scan_type = st.selectbox(
            "Select Scan Type",
            ["Bullish (Upward Momentum)", "Bearish (Downward Momentum)", "Breakouts", "Unusual Volume"],
            key="scan_type_tab2"
        )
        max_results = st.slider("Max Stocks to Display", 1, 200, 20, key="max_results_tab2")
        
        # Botón para iniciar el escaneo
        if st.button("🚀 Run Market Scan", key="run_scan_tab2"):
            with st.spinner(f"Scanning Market ({scan_type})..."):
                # Obtener lista de stocks a escanear
                stocks_to_scan = get_metaverse_stocks()
                st.write(f"Scanning {len(stocks_to_scan)} stocks: {stocks_to_scan[:25]}...")
                
                # Inicializar estructuras para almacenar datos
                scan_data = []
                alerts = []
                failed_stocks = []
                extra_metrics = {"avg_iv": 0, "max_gwe": 0, "key_levels": []}
                
                # Obtener precios actuales
                prices_dict = get_current_prices(stocks_to_scan)
                
                # Escanear stocks en paralelo
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(get_historical_prices_combined, stock, limit=30): stock
                        for stock in stocks_to_scan
                    }
                    for future in futures:
                        stock = futures[future]
                        try:
                            # Obtener precio actual
                            current_price = prices_dict.get(stock, 0.0)
                            if not current_price or current_price <= 0:
                                current_price = 1.0
                                logger.debug(f"{stock}: No valid current price, using fallback $1.0")
                            
                            # Obtener precios y volúmenes históricos
                            prices, volumes = future.result()
                            if not prices or len(prices) < 5:
                                prices = [current_price] * 10
                                volumes = [1000000] * 10
                                logger.debug(f"{stock}: Insufficient historical data, using fallback.")
                            
                            # Convertir a arrays de numpy
                            prices = np.array(prices)
                            volumes = np.array(volumes)
                            returns = np.diff(prices) / prices[:-1]
                            vol_historical = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.1
                            
                            # Obtener datos de opciones
                            exp_dates = get_expiration_dates(stock)
                            iv, gwe, skew_dynamic = vol_historical, 0, 0
                            support_level, resistance_level = current_price * 0.95, current_price * 1.05
                            if exp_dates:
                                options_data = get_options_data(stock, exp_dates[0])
                                if options_data:
                                    iv_calls = np.mean([
                                        float(opt["greeks"].get("smv_vol", 0))
                                        for opt in options_data
                                        if opt.get("option_type", "").lower() == "call" and "greeks" in opt
                                    ]) or vol_historical
                                    iv_puts = np.mean([
                                        float(opt["greeks"].get("smv_vol", 0))
                                        for opt in options_data
                                        if opt.get("option_type", "").lower() == "put" and "greeks" in opt
                                    ]) or vol_historical
                                    iv = np.mean([iv_calls, iv_puts])
                                    extra_metrics["avg_iv"] += iv
                                    oi_calls = sum(int(opt.get("open_interest", 0)) for opt in options_data if opt.get("option_type", "").lower() == "call")
                                    oi_puts = sum(int(opt.get("open_interest", 0)) for opt in options_data if opt.get("option_type", "").lower() == "put")
                                    skew_dynamic = (iv_calls - iv_puts) * (oi_calls / (oi_puts + 1)) / iv if iv > 0 else 0
                                    strikes = np.array([float(opt["strike"]) for opt in options_data])
                                    strikes_near_price = strikes[np.abs(strikes - current_price) < current_price * 0.1]
                                    gwe = sum(
                                        float(opt["greeks"].get("gamma", 0)) * int(opt.get("open_interest", 0)) / (abs(float(opt["strike"]) - current_price) + 0.01)
                                        for opt in options_data
                                        if "greeks" in opt and float(opt["strike"]) in strikes_near_price
                                    )
                                    gwe *= current_price / 1000 if gwe != 0 else 0
                                    extra_metrics["max_gwe"] = max(extra_metrics["max_gwe"], abs(gwe))
                                    support_level = np.min(strikes) if strikes.size > 0 else current_price * 0.95
                                    resistance_level = np.max(strikes) if strikes.size > 0 else current_price * 1.05
                                    extra_metrics["key_levels"].append({"stock": stock, "support": support_level, "resistance": resistance_level})
                            
                            # Calcular métricas
                            volume_avg = np.mean(volumes[:-5]) if len(volumes) > 5 else 1
                            volume_spike = np.max(volumes[-5:]) / volume_avg if volume_avg > 0 else 1.0
                            oi_total = sum(int(opt.get("open_interest", 0)) for opt in options_data) if exp_dates and options_data else 0
                            lmi = volume_spike * (1 + oi_total / (1000000 * vol_historical + 1)) * (1 / (vol_historical + 0.1))
                            
                            momentum = calculate_momentum(prices, vol_historical)
                            rsi = calculate_rsi(prices)
                            
                            # Puntaje de catalizadores
                            catalyst_score = 0
                            url = f"{FMP_BASE_URL}/earning_calendar"
                            params = {"apikey": FMP_API_KEY, "from": datetime.now().strftime('%Y-%m-%d'), "to": (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d')}
                            try:
                                response = session_fmp.get(url, params=params, headers=HEADERS_FMP, timeout=5)
                                response.raise_for_status()
                                earnings_data = response.json()
                                if earnings_data and any(e.get("symbol") == stock for e in earnings_data):
                                    catalyst_score += 40 if iv > vol_historical * 1.5 else 25
                            except Exception as e:
                                logger.error(f"FMP Earnings error for {stock}: {str(e)}")
                            
                            url_macro = f"{FMP_BASE_URL}/economic-calendar"
                            params_macro = {"apikey": FMP_API_KEY, "from": datetime.now().strftime('%Y-%m-%d'), "to": (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')}
                            try:
                                response = session_fmp.get(url_macro, params=params_macro, headers=HEADERS_FMP, timeout=5)
                                response.raise_for_status()
                                macro_data = response.json()
                                if macro_data and len(macro_data) > 0:
                                    catalyst_score += 30 if any(e.get("impact", "Low") in ["High", "Medium"] for e in macro_data) else 15
                            except Exception as e:
                                logger.error(f"FMP Macro error for {stock}: {str(e)}")
                            
                            # Calcular FMS (Future Motion Score)
                            iv_hv_ratio = iv / vol_historical if vol_historical > 0 else 1.0
                            iv_weight = 40 + (iv_hv_ratio - 1) * 10 if iv_hv_ratio > 1 else 40
                            gwe_weight = 35 + abs(gwe) * 5 if abs(gwe) > 0.5 else 35
                            lmi_weight = 25 + (lmi - 1) * 5 if lmi > 1 else 25
                            skew_weight = 20 + abs(skew_dynamic) * 10 if abs(skew_dynamic) > 0.2 else 20
                            momentum_weight = 15 + abs(momentum) * 5 if abs(momentum) > 0.1 else 15
                            fms = iv_hv_ratio * iv_weight + abs(gwe) * gwe_weight + lmi * lmi_weight + abs(skew_dynamic) * skew_weight + abs(momentum) * momentum_weight + catalyst_score
                            
                            # Determinar dirección con ajuste para Bearish
                            direction_score = (gwe * 0.5) + (skew_dynamic * 0.3) + (momentum * 0.2)
                            direction = "Up" if direction_score > 0.7 and (rsi < 35 or lmi > 2.5) else "Down" if direction_score < -0.3 and (rsi > 50 or lmi > 1.5) else "Neutral"
                            
                            # Filtrar según tipo de escaneo
                            if scan_type == "Bullish (Upward Momentum)" and (direction != "Up" or fms < 100):
                                continue
                            elif scan_type == "Bearish (Downward Momentum)" and (direction != "Down" or fms < 25):  # Ajustado de 50 a 25
                                continue
                            elif scan_type == "Breakouts" and abs(direction_score) < 0.9:
                                continue
                            elif scan_type == "Unusual Volume" and lmi < 2.0:
                                continue
                            
                            # Calcular GCF (Confidence Factor)
                            signal_strength = min(1.0, abs(direction_score) / 2.0) * 50
                            catalyst_boost = catalyst_score * 1.5
                            agreement = 30 if (gwe * skew_dynamic > 0 and gwe * momentum > 0) else 15
                            liquidity_boost = 20 if lmi > 2.0 and abs(rsi - 50) < 20 else 0
                            gcf = min(100, signal_strength + catalyst_boost + agreement + liquidity_boost)
                            
                            # Generar alertas
                            if gcf > 95 and fms > 150:
                                alerts.append(f"⚠️ HIGH CONFIDENCE ALERT: {stock} | FMS: {fms:.1f} | Direction: {direction} | GCF: {gcf:.1f}%")
                            
                            # Almacenar datos
                            scan_data.append({
                                "Ticker": stock,
                                "Price": current_price,
                                "IV/HV": iv_hv_ratio,
                                "GWE": gwe,
                                "Skew": skew_dynamic,
                                "LMI": lmi,
                                "Momentum": momentum,
                                "RSI": rsi,
                                "FMS": fms,
                                "Direction": direction,
                                "GCF": gcf,
                                "Catalyst": catalyst_score > 0,
                                "Support": support_level,
                                "Resistance": resistance_level,
                                "Volume": volumes[-1] if volumes.size > 0 else 0
                            })
                        except Exception as e:
                            logger.error(f"Error scanning {stock}: {str(e)}")
                            failed_stocks.append((stock, str(e)))
                
                # Mostrar resultados
                if scan_data:
                    df_scan = pd.DataFrame(scan_data).sort_values("FMS", ascending=False)[:max_results]
                    styled_df = df_scan.style.format({
                        "Price": "${:.2f}",
                        "IV/HV": "{:.2f}",
                        "GWE": "{:.2f}",
                        "Skew": "{:.2f}",
                        "LMI": "{:.2f}",
                        "Momentum": "{:.2f}",
                        "RSI": "{:.1f}",
                        "FMS": "{:.1f}",
                        "GCF": "{:.1f}",
                        "Support": "${:.2f}",
                        "Resistance": "${:.2f}",
                        "Volume": "{:,.0f}"
                    }).background_gradient(cmap="Purples", subset=["FMS"]).background_gradient(cmap="Greens", subset=["GCF"])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Mostrar alertas
                    if alerts:
                        st.warning("\n".join(alerts))
                    
                    # Mostrar el mejor resultado solo si hay datos
                    if not df_scan.empty:
                        top_pick = df_scan.iloc[0]
                        st.success(f"Top Pick: {top_pick['Ticker']} | FMS: {top_pick['FMS']:.1f} | Direction: {top_pick['Direction']} | GCF: {top_pick['GCF']:.1f}%")
                        
                        # Crear gráfica
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=df_scan["Ticker"], y=df_scan["FMS"], name="Future Motion Score", marker_color="purple"))
                        fig.add_trace(go.Scatter(x=df_scan["Ticker"], y=df_scan["GCF"], name="Confidence", mode="lines+markers", yaxis="y2", line=dict(color="lime")))
                        fig.add_trace(go.Scatter(x=df_scan["Ticker"], y=df_scan["Support"], name="Support", mode="lines", line=dict(color="cyan", dash="dash")))
                        fig.add_trace(go.Scatter(x=df_scan["Ticker"], y=df_scan["Resistance"], name="Resistance", mode="lines", line=dict(color="red", dash="dash")))
                        fig.add_trace(go.Bar(x=df_scan["Ticker"], y=df_scan["Volume"], name="Volume", marker_color="blue", opacity=0.5, yaxis="y3"))
                        fig.update_layout(
                            xaxis_title="Ticker",
                            yaxis_title="Future Motion Score (FMS)",
                            yaxis2=dict(title="Confidence Factor (GCF %)", overlaying="y", side="right", range=[0, 100]),
                            yaxis3=dict(title="Volume", overlaying="y", side="left", anchor="free", position=0.05, range=[0, max(df_scan["Volume"]) * 1.2] if not df_scan["Volume"].empty else [0, 1000000]),
                            template="plotly_dark",
                            plot_bgcolor="#1E1E1E",
                            paper_bgcolor="#1E1E1E",
                            font=dict(color="#FFFFFF", size=14),
                            legend=dict(yanchor="top", y=1.1, xanchor="right", x=1),
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Descarga de datos
                        csv_scan = df_scan.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Market Scan Data",
                            data=csv_scan,
                            file_name=f"market_scan_pro_{scan_type.replace(' ', '_').lower()}.csv",
                            mime="text/csv",
                            key="download_scan_tab2"
                        )
                        
                        # Insights adicionales
                        st.markdown("#### Extra Scan Insights")
                        num_stocks = len(scan_data)
                        extra_metrics["avg_iv"] = extra_metrics["avg_iv"] / num_stocks if num_stocks > 0 else 0
                        st.write(f"**Average Implied Volatility (IV):** {extra_metrics['avg_iv']:.2%}")
                        st.write(f"**Max Gamma Weighted Exposure (GWE):** {extra_metrics['max_gwe']:.2f}")
                        st.write("**Key Levels (Top 5 Stocks by FMS):**")
                        top_5_levels = sorted(extra_metrics["key_levels"], key=lambda x: df_scan[df_scan["Ticker"] == x["stock"]]["FMS"].iloc[0] if x["stock"] in df_scan["Ticker"].values else 0, reverse=True)[:5]
                        for level in top_5_levels:
                            st.write(f"- {level['stock']}: Support: ${level['support']:.2f}, Resistance: ${level['resistance']:.2f}")
                    else:
                        st.warning(f"No stocks met the criteria for '{scan_type}'. Try adjusting the scan type or check data availability.")
                else:
                    st.error("")
                    if failed_stocks:
                        st.write("Failed stocks and reasons:")
                        for stock, reason in failed_stocks[:11]:
                            st.write(f"- {stock}: {reason}")
                    st.write("Stocks attempted:", stocks_to_scan[:25])
                    
                
                # Pie de página
                st.markdown(f"*Last Scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by Ozy*")
    # Tab 3: News Scanner
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
                st.markdown("---")
                st.markdown("*Developed by Ozy | © 2025*")

    # Tab 4: Institutional Holders
    with tab4:
        st.subheader("Institutional Holders")
        ticker = st.text_input("Ticker for Holders (e.g., AAPL):", "AAPL", key="holders_ticker").upper()
        if ticker:
            holders = get_institutional_holders_list(ticker)
            if holders is not None and not holders.empty:
                def color_negative(row):
                    if 'Change' in row and row['Change'] < 0:
                        return ['color: #FF4500'] * len(row)
                    elif 'Shares' in row and row['Shares'] < 0:
                        return ['color: #FF4500'] * len(row)
                    return [''] * len(row)
                styled_holders = holders.style.apply(color_negative, axis=1).format({
                    'Shares': '{:,.0f}',
                    'Change': '{:,.0f}' if 'Change' in holders.columns else None,
                    'Value': '${:,.0f}' if 'Value' in holders.columns else None
                })
                st.dataframe(styled_holders, use_container_width=True)
                holders_csv = holders.to_csv(index=False)
                st.download_button(
                    label="📥 Download Holders Data",
                    data=holders_csv,
                    file_name=f"{ticker}_institutional_holders.csv",
                    mime="text/csv",
                    key="download_tab4"
                )
            else:
                st.warning("No institutional holders data available.")
                st.markdown("---")
                st.markdown("*Developed by Ozy | © 2025*")

    # Tab 5: Options Order Flow
    with tab5:
        st.subheader("Order Flow")
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
                    st.write(f"- **ROE**: {financial_metrics.get('ROE', 0):,.2f}")
                    st.write(f"- **Beta**: ${financial_metrics.get('Beta', 0):,.2f}")
                    st.write(f"- **PE Ratio**: ${financial_metrics.get('PE Ratio', 0):,.2f}")
                    st.write(f"- **Debt-to-Equity Ratio**: {financial_metrics.get('Debt-to-Equity Ratio', 0):.2f}")
                    st.write(f"- **Market Cap**: ${financial_metrics.get('Market Cap', 0):,.2f}")
                    st.write(f"- **Operating Cash Flow**: ${financial_metrics.get('Operating Cash Flow', 0):,.2f}")
                    st.write(f"- **Free Cash Flow**: ${financial_metrics.get('Free Cash Flow', 0):,.2f}")
                    st.markdown(f"### Possibilities for {stock}")
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
                        buy_calls_data = pd.DataFrame({"strike": all_strikes}).merge(buy_calls[["strike", "open_interest", "volume"]], on="strike", how="left").fillna({"open_interest": 0, "volume": 0})
                        sell_calls_data = pd.DataFrame({"strike": all_strikes}).merge(sell_calls[["strike", "open_interest", "volume"]], on="strike", how="left").fillna({"open_interest": 0, "volume": 0})
                        buy_puts_data = pd.DataFrame({"strike": all_strikes}).merge(buy_puts[["strike", "open_interest", "volume"]], on="strike", how="left").fillna({"open_interest": 0, "volume": 0})
                        sell_puts_data = pd.DataFrame({"strike": all_strikes}).merge(sell_puts[["strike", "open_interest", "volume"]], on="strike", how="left").fillna({"open_interest": 0, "volume": 0})
                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            x=buy_calls_data["strike"],
                            y=buy_calls_data["open_interest"],
                            name="Buy Call",
                            marker_color="green",
                            width=0.4,
                            hovertemplate="%{y:,}",
                            text="Buy Call",
                            textposition="inside"
                        ))
                        fig.add_trace(go.Bar(
                            x=sell_calls_data["strike"],
                            y=sell_calls_data["open_interest"],
                            name="Sell Call",
                            marker_color="orange",
                            width=0.4,
                            hovertemplate="%{y:,}",
                            text="Sell Call",
                            textposition="inside"
                        ))
                        fig.add_trace(go.Bar(
                            x=buy_puts_data["strike"],
                            y=-buy_puts_data["open_interest"],
                            name="Buy Put",
                            marker_color="red",
                            width=0.4,
                            hovertemplate="%{customdata:,}",
                            customdata=[abs(oi) for oi in buy_puts_data["open_interest"]],
                            text="Buy Put",
                            textposition="inside"
                        ))
                        fig.add_trace(go.Bar(
                            x=sell_puts_data["strike"],
                            y=-sell_puts_data["open_interest"],
                            name="Sell Put",
                            marker_color="purple",
                            width=0.4,
                            hovertemplate="%{customdata:,}",
                            customdata=[abs(oi) for oi in sell_puts_data["open_interest"]],
                            text="Sell Put",
                            textposition="inside"
                        ))

                        fig.update_traces(
                            hoverlabel=dict(
                                bgcolor="rgba(0,0,0,0.1)",
                                bordercolor="rgba(255,255,255,0.3)",
                                font=dict(color="white", size=12)
                            )
                        )

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

                        y_max = max(buy_calls_data["open_interest"].max() or 0, sell_calls_data["open_interest"].max() or 0) * 1.1 or 100
                        y_min = -y_max
                        fig.add_trace(go.Scatter(
                            x=[current_price, current_price],
                            y=[y_min, y_max],
                            mode="lines",
                            line=dict(width=1, dash="dash", color="#39FF14"),
                            name="Current Price",
                            hovertemplate=(
                                f"<b>Current Price:</b> ${current_price:.2f}<br>"
                                f"<b>Max Pain:</b> ${max_pain:.2f}<br>"
                                f"<b>Total Call OI:</b> {total_call_oi:,}<br>"
                                f"<b>Total Put OI:</b> {total_put_oi:,}<br>"
                                f"<b>Net Gamma:</b> {net_gamma:.2f}<br>"
                                f"<b>MM Direction:</b> {direction_mm}"
                            ),
                            showlegend=False,
                            hoverlabel=dict(
                                bgcolor="rgba(0,0,0,0)",
                                bordercolor="rgba(0,0,0,0)",
                                font=dict(color="#39FF14", size=12)
                            )
                        ))

                        fig.add_annotation(
                            x=current_price,
                            y=y_max * 0.95,
                            text=f"Price: ${current_price:.2f}",
                            showarrow=False,
                            font=dict(color="#39FF14", size=10),
                            bgcolor="rgba(0,0,0,0.5)",
                            bordercolor="#39FF14",
                            borderwidth=1,
                            borderpad=4
                        )

                        score_color = "red" if combined_score < 0 else "green" if combined_score > 0 else "white"
                        fig.add_annotation(
                            x=current_price,
                            y=y_max * 0.9,
                            text=f"MM Score: <span style='color:{score_color}'>{combined_score:.2f}</span><br>{direction_mm}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor="#39FF14",
                            font=dict(size=12),
                            ax=20,
                            ay=-30,
                            bgcolor="rgba(0,0,0,0.5)",
                            bordercolor="#39FF14"
                        )

                        fig.update_layout(
                            title=f"| {stock} |  Expiration: {selected_expiration}",
                            xaxis_title="Strike Price",
                            yaxis_title="Open Interest",
                            barmode="relative",
                            showlegend=True,
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                            xaxis=dict(
                                tickmode="array",
                                tickvals=all_strikes,
                                ticktext=[f"{s:.2f}" for s in all_strikes],
                                range=[min(all_strikes) - 10, max(all_strikes) + 10],
                                showgrid=False,
                                rangeslider=dict(visible=True)
                            ),
                            yaxis=dict(showgrid=False),
                            hovermode="x",
                            bargap=0.2,
                            height=600,
                            template="plotly_dark"
                        )

                        st.plotly_chart(fig, use_container_width=True, height=600)
                        order_flow_df = pd.DataFrame({
                            "Strike": all_strikes,
                            "Buy_Call_OI": buy_calls_data["open_interest"],
                            "Sell_Call_OI": sell_calls_data["open_interest"],
                            "Buy_Put_OI": buy_puts_data["open_interest"],
                            "Sell_Put_OI": sell_puts_data["open_interest"],
                            "Max_Pain": [max_pain] * len(all_strikes),
                            "Total_Call_OI": [total_call_oi] * len(all_strikes),
                            "Total_Put_OI": [total_put_oi] * len(all_strikes),
                            "Net_Gamma": [net_gamma] * len(all_strikes),
                            "MM_Direction": [direction_mm] * len(all_strikes)
                        })
                        order_flow_csv = order_flow_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Order Flow Data",
                            data=order_flow_csv,
                            file_name=f"{stock}_order_flow_{selected_expiration}.csv",
                            mime="text/csv",
                            key="download_tab5"
                        )
                        st.markdown("---")
                        st.markdown("*Developed by Ozy | © 2025*")

    # Tab 6: Analyst Rating Flow
        # Tab 6: Analyst Rating Flow
    with tab6:
        st.subheader("Rating Flow")
        col1, col2 = st.columns([1, 3])
        
        # Estilos personalizados
        st.markdown("""
            <style>
            .rating-flow-container {
                background: linear-gradient(135deg, #1E1E1E, #2A2A2A);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
            }
            .tooltip {
                position: relative;
                display: inline-block;
                cursor: help;
                color: #32CD32;
                margin-left: 5px;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 200px;
                background-color: #2D2D2D;
                color: #FFFFFF;
                text-align: center;
                border-radius: 5px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                font-size: 12px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
            }
            .hacker-text {
                background: #1A1A1A;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #FFD700;
                font-family: 'Courier New', Courier, monospace;
                color: #FFD700;
                font-size: 14px;
                line-height: 1.5;
                text-align: left;
                box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
                white-space: pre-wrap;
            }
            </style>
        """, unsafe_allow_html=True)

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
            
            # Calcular volatilidad implícita
            iv = get_implied_volatility(ticker) or 0.3
            iv_factor = min(max(iv, 0.1), 1.0)
            
            # Vol Filter
            st.markdown("### Vol Filter")
            volume_options = {
                "0.1M": 10000,
                "0.2M": 20000,
                "0.3M": 30000,
                "0.4M": 40000,
                "0.5M": 50000,
                "1.0M": 100000
            }
            auto_oi = int(100000 * (1 + iv_factor * 2))
            auto_oi_key = next((k for k, v in volume_options.items() if v >= auto_oi), "0.1M")
            use_auto_oi = st.checkbox("Auto OI (Volatility-Based)", value=False, key="auto_oi")
            if use_auto_oi:
                open_interest_threshold = volume_options[auto_oi_key]
                st.write(f"Auto OI Set: {auto_oi_key} ({volume_options[auto_oi_key]:,})")
            else:
                selected_volume = st.selectbox("Min Open Interest (M)", list(volume_options.keys()), index=0, key="alerts_vol")
                open_interest_threshold = volume_options[selected_volume]
            
            # Gamma Filter
            st.markdown("### Gamma Filter")
            gamma_options = {
                "0.001": 0.001,
                "0.005": 0.005,
                "0.01": 0.01,
                "0.02": 0.02,
                "0.03": 0.03,
                "0.05": 0.05
            }
            auto_gamma = max(0.001, min(0.05, iv_factor / 20))
            auto_gamma_key = next((k for k, v in gamma_options.items() if v >= auto_gamma), "0.001")
            use_auto_gamma = st.checkbox("Auto Gamma (Volatility-Based)", value=False, key="auto_gamma")
            if use_auto_gamma:
                gamma_threshold = gamma_options[auto_gamma_key]
                st.write(f"Auto Gamma Set: {auto_gamma_key}")
            else:
                selected_gamma = st.selectbox("Min Gamma", list(gamma_options.keys()), index=0, key="alerts_gamma")
                gamma_threshold = gamma_options[selected_gamma]
            
            st.markdown(f"**Current Price:** ${current_price:.2f}  \n*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        with col2:
            with st.spinner(f"Generating alerts for {expiration_date}..."):
                options_data = get_options_data(ticker, expiration_date)
                if not options_data:
                    st.error("No options data available for this date.")
                    return
                
                # Calcular Max Pain con suma de pérdidas y ganancia del MM
                strikes = sorted(set(float(opt["strike"]) for opt in options_data))
                min_loss = float('inf')
                max_pain = None
                call_loss_at_max_pain = 0
                put_loss_at_max_pain = 0
                for test_strike in strikes:
                    call_loss = sum(max(0, float(opt["strike"]) - test_strike) * int(opt["open_interest"]) 
                                    for opt in options_data if opt.get("option_type", "").upper() == "CALL")
                    put_loss = sum(max(0, test_strike - float(opt["strike"])) * int(opt["open_interest"]) 
                                   for opt in options_data if opt.get("option_type", "").upper() == "PUT")
                    total_loss = call_loss + put_loss
                    if total_loss < min_loss:
                        min_loss = total_loss
                        max_pain = test_strike
                        call_loss_at_max_pain = call_loss
                        put_loss_at_max_pain = put_loss
                
                # Ganancia del MM en dólares
                mm_gain = (call_loss_at_max_pain + put_loss_at_max_pain) * 100
                
                # Texto estilo hacker
                hacker_text = f"""
>>> Current_Price = ${current_price:.2f}
>>> Updated = "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
>>> Max_Pain_Strike = ${max_pain:.2f}
>>> CALL_Loss = ${call_loss_at_max_pain * 100:,.2f}
>>> PUT_Loss = ${put_loss_at_max_pain * 100:,.2f}
>>> MM_Potential_Gain = ${mm_gain:,.2f}
"""
                st.markdown(f'<div class="hacker-text">{hacker_text}</div>', unsafe_allow_html=True)

                # Función para generar sugerencias
                def generate_contract_suggestions(ticker, options_data, current_price, oi_threshold, gamma_threshold, max_pain, mm_gain):
                    suggestions = []
                    valid_contracts = 0
                    for opt in options_data:
                        if not isinstance(opt, dict):
                            continue
                        strike = float(opt.get("strike", 0))
                        opt_type = opt.get("option_type", "").upper()
                        oi = int(opt.get("open_interest", 0))
                        greeks = opt.get("greeks", {})
                        gamma = float(greeks.get("gamma", 0)) if isinstance(greeks, dict) else 0
                        iv = float(greeks.get("smv_vol", 0)) if isinstance(greeks, dict) else 0
                        delta = float(greeks.get("delta", 0)) if isinstance(greeks, dict) else 0
                        volume = int(opt.get("volume", 0))
                        last = opt.get("last")
                        bid = opt.get("bid")
                        last_price = float(last) if last is not None and isinstance(last, (int, float, str)) else float(bid) if bid is not None and isinstance(bid, (int, float, str)) else 0
                        
                        if oi >= oi_threshold and gamma >= gamma_threshold:
                            valid_contracts += 1
                            buy_ratio = 0.5 + iv_factor * 0.2 if opt_type == "CALL" else 0.5 - iv_factor * 0.2
                            buy_volume = int(volume * buy_ratio)
                            sell_volume = volume - buy_volume
                            avg_buy_price = last_price * (1 + iv_factor * 0.02) if last_price else 0
                            avg_sell_price = last_price * (1 - iv_factor * 0.02) if last_price else 0
                            total_buy = buy_volume * avg_buy_price * 100
                            total_sell = sell_volume * avg_sell_price * 100
                            action = "SELL" if (opt_type == "CALL" and strike > current_price) or (opt_type == "PUT" and strike < current_price) else "BUY"
                            rr = abs(strike - current_price) / (last_price + 0.01) if last_price else 0
                            prob_otm = 1 - abs(delta) if delta else 0
                            profit = (strike - current_price) * 100 if action == "BUY" and opt_type == "CALL" else (current_price - strike) * 100 if action == "BUY" and opt_type == "PUT" else last_price * 100
                            is_max_pain = strike == max_pain
                            mm_gain_at_strike = mm_gain if is_max_pain else 0
                            
                            suggestions.append({
                                "Strike": strike, "Action": action, "Type": opt_type, "Gamma": gamma, "IV": iv, "Delta": delta,
                                "RR": rr, "Prob OTM": prob_otm, "Profit": profit, "Open Interest": oi, "IsMaxPain": is_max_pain,
                                "Buy Volume": buy_volume, "Sell Volume": sell_volume, "Avg Buy Price": avg_buy_price,
                                "Avg Sell Price": avg_sell_price, "Total Buy ($)": total_buy, "Total Sell ($)": total_sell,
                                "MM Gain ($)": mm_gain_at_strike
                            })
                    
                    st.write(f"Max Found {valid_contracts} contracts  >= {oi_threshold:,} and GM >= {gamma_threshold}")
                    return suggestions

                suggestions = generate_contract_suggestions(ticker, options_data, current_price, open_interest_threshold, gamma_threshold, max_pain, mm_gain)
                if suggestions:
                    df = pd.DataFrame(suggestions)
                    df['Contract'] = df.apply(lambda row: f"{ticker} {row['Action']} {row['Type']} {row['Strike']}", axis=1)
                    df = df[['Contract', 'Strike', 'Action', 'Type', 'Gamma', 'IV', 'Delta', 'RR', 'Prob OTM', 'Profit', 'Open Interest', 
                             'Buy Volume', 'Sell Volume', 'Avg Buy Price', 'Avg Sell Price', 'Total Buy ($)', 'Total Sell ($)', 'MM Gain ($)', 'IsMaxPain']]
                    df.columns = ['Contract', 'Strike', 'Action', 'Type', 'Gamma', 'IV', 'Delta', 'R/R', 'Prob OTM', 'Profit ($)', 'Open Int.', 
                                  'Buy Vol.', 'Sell Vol.', 'Avg Buy ($)', 'Avg Sell ($)', 'Total Buy ($)', 'Total Sell ($)', 'MM Gain ($)', 'Max Pain']

                    def color_row(row):
                        if row['Max Pain']:
                            return ['color: #FFD700'] * len(row)
                        elif row['Type'] == "CALL":
                            return ['color: #228B22' if row['Action'] == "SELL" and row['Strike'] > current_price else 'color: #006400'] * len(row)
                        elif row['Type'] == "PUT":
                            return ['color: #CD5C5C' if row['Action'] == "SELL" and row['Strike'] < current_price else 'color: #8B0000'] * len(row)
                        return [''] * len(row)

                    styled_df = df.style.apply(color_row, axis=1).format({
                        'Strike': '{:.1f}', 'Gamma': '{:.4f}', 'IV': '{:.2f}', 'Delta': '{:.2f}', 'R/R': '{:.2f}',
                        'Prob OTM': '{:.2%}', 'Profit ($)': '${:.2f}', 'Open Int.': '{:,.0f}',
                        'Buy Vol.': '{:,.0f}', 'Sell Vol.': '{:,.0f}', 'Avg Buy ($)': '${:.2f}', 'Avg Sell ($)': '${:.2f}',
                        'Total Buy ($)': '${:,.2f}', 'Total Sell ($)': '${:,.2f}', 'MM Gain ($)': lambda x: '${:,.2f}'.format(x) if x > 0 else '-'
                    })
                    st.dataframe(styled_df, height=400)
                    
                    # Gráfico de burbujas
                    fig = go.Figure()
                    call_df = df[df['Type'] == 'CALL']
                    if not call_df.empty:
                        fig.add_trace(go.Scatter(x=call_df['Strike'], 
                                               y=call_df['Avg Buy ($)'], 
                                               mode='markers', name='CALL Buy ($)', 
                                               marker=dict(size=call_df['Total Buy ($)'] / call_df['Total Buy ($)'].max() * 50, 
                                                         color='#228B22', opacity=0.7),
                                               text=call_df['Contract'] + '<br>Total: $' + call_df['Total Buy ($)'].astype(str)))
                    put_df = df[df['Type'] == 'PUT']
                    if not put_df.empty:
                        fig.add_trace(go.Scatter(x=put_df['Strike'], 
                                               y=-put_df['Avg Sell ($)'], 
                                               mode='markers', name='PUT Sell ($)', 
                                               marker=dict(size=put_df['Total Sell ($)'] / put_df['Total Sell ($)'].max() * 50, 
                                                         color='#CD5C5C', opacity=0.7),
                                               text=put_df['Contract'] + '<br>Total: $' + put_df['Total Sell ($)'].astype(str)))
                    if mm_gain > 0:
                        fig.add_trace(go.Scatter(x=[max_pain], 
                                               y=[0], 
                                               mode='markers+text', name='MM Gain', 
                                               marker=dict(size=50, color='#FFD700', opacity=0.9, 
                                                         line=dict(width=2, color='#FFFFFF')),
                                               text=[f"MM Gain: ${mm_gain:,.2f}"],
                                               textposition="middle center"))
                    fig.add_vline(x=max_pain, line=dict(color="#FFD700", width=2, dash="dash"), annotation_text="Max Pain", annotation_position="top right")
                    fig.add_hline(y=0, line=dict(color="#FFFFFF", width=1))
                    fig.update_layout(title="AverageS", 
                                    xaxis_title="Strike", yaxis_title="Avg Price ($)", 
                                    template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Rating Flow Data",
                        data=csv,
                        file_name=f"{ticker}_rating_flow_{expiration_date}.csv",
                        mime="text/csv",
                        key="download_tab6"
                    )
                else:
                    st.error(f"No alerts generated with Open Interest ≥ {open_interest_threshold:,}, Gamma ≥ {gamma_threshold}. Check logs.")
                
                st.markdown("---")
                st.markdown("*Developed by Ozy | © 2025*")

    # Tab 7: Elliott Pulse
    with tab7:
        st.subheader("Elliott Pulse")
        ticker = st.text_input("Ticker Symbol (e.g., SPY)", "SPY", key="elliott_ticker").upper()
        expiration_dates = get_expiration_dates(ticker)
        if not expiration_dates:
            st.error(f"No expiration dates found for '{ticker}'. Try a valid ticker (e.g., SPY).")
            return
        selected_expiration = st.selectbox("Select Expiration Date", expiration_dates, key="elliott_exp_date")

        with st.spinner(f"Fetching data for {ticker} on {selected_expiration}..."):
            current_price = get_current_price(ticker)
            if current_price == 0.0:
                st.error(f"Unable to fetch current price for '{ticker}'.")
                return
            options_data = get_options_data(ticker, selected_expiration)
            if not options_data:
                st.error(f"No options data available for {selected_expiration}.")
                return

            total_oi_all = sum(int(opt.get("open_interest", 0)) for opt in options_data)
            num_strikes = len(set(opt.get("strike", 0) for opt in options_data))
            avg_oi = total_oi_all / num_strikes if num_strikes > 0 else 0
            st.markdown(f"**Avg OI per Strike:** {avg_oi:,.0f}")

            default_volume = max(0.0001, min(5.0, avg_oi / 2_000_000))
            volume_threshold = st.slider("Min Open Interest (millions)", 0.0001, 5.0, default_volume, step=0.0001, key="elliott_vol") * 1_000_000

            strikes_data = {}
            for opt in options_data:
                strike = float(opt.get("strike", 0))
                opt_type = opt.get("option_type", "").upper()
                oi = int(opt.get("open_interest", 0))
                greeks = opt.get("greeks", {})
                gamma = float(greeks.get("gamma", 0)) if isinstance(greeks, dict) else 0
                if strike not in strikes_data:
                    strikes_data[strike] = {"CALL": {"OI": 0, "Gamma": 0}, "PUT": {"OI": 0, "Gamma": 0}}
                strikes_data[strike][opt_type]["OI"] += oi
                strikes_data[strike][opt_type]["Gamma"] += gamma * oi

            strikes = sorted(strikes_data.keys())
            call_gamma = []
            put_gamma = []
            net_gamma = []
            total_oi = []
            for strike in strikes:
                call_oi = strikes_data[strike]["CALL"]["OI"]
                put_oi = strikes_data[strike]["PUT"]["OI"]
                if call_oi >= volume_threshold or put_oi >= volume_threshold:
                    cg = strikes_data[strike]["CALL"]["Gamma"]
                    pg = strikes_data[strike]["PUT"]["Gamma"]
                    call_gamma.append(cg)
                    put_gamma.append(-pg)
                    net_gamma.append(cg - pg)
                    total_oi.append(call_oi + put_oi)
                else:
                    call_gamma.append(0)
                    put_gamma.append(0)
                    net_gamma.append(0)
                    total_oi.append(0)

            significant_strikes = [(strike, ng, oi) for strike, ng, oi in zip(strikes, net_gamma, total_oi) if oi > volume_threshold]
            if not significant_strikes:
                st.warning(f"No significant strikes found above volume threshold ({volume_threshold/1_000_000:.4f}M).")
                return

            total_volume = sum(oi for _, _, oi in significant_strikes)
            volume_cutoff = total_volume * 0.1
            high_volume_strikes = [(strike, oi) for strike, _, oi in significant_strikes if oi >= volume_cutoff]

            max_pain = None
            min_loss = float('inf')
            for strike in [s[0] for s in high_volume_strikes]:
                call_loss = sum(max(0, s - strike) * strikes_data[s]["CALL"]["OI"] for s, _ in high_volume_strikes)
                put_loss = sum(max(0, strike - s) * strikes_data[s]["PUT"]["OI"] for s, _ in high_volume_strikes)
                total_loss = call_loss + put_loss
                if total_loss < min_loss:
                    min_loss = total_loss
                    max_pain = strike
            max_pain_gamma = strikes_data.get(max_pain, {"CALL": {"Gamma": 0}, "PUT": {"Gamma": 0}})["CALL"]["Gamma"] - \
                            strikes_data.get(max_pain, {"CALL": {"Gamma": 0}, "PUT": {"Gamma": 0}})["PUT"]["Gamma"] if max_pain else 0

            sorted_by_volume = sorted(significant_strikes, key=lambda x: x[2], reverse=True)
            sorted_by_low_volume = sorted(significant_strikes, key=lambda x: x[2])

            a_point = min(significant_strikes, key=lambda x: abs(x[0] - current_price)) if significant_strikes else (current_price, 0, 0)
            b_point = sorted_by_volume[0] if sorted_by_volume else (current_price + 1, 0, 0)
            c_point = sorted_by_low_volume[0] if sorted_by_low_volume else (current_price - 1, 0, 0)
            d_point = (max_pain, max_pain_gamma, strikes_data.get(max_pain, {"CALL": {"OI": 0}, "PUT": {"OI": 0}})["CALL"]["OI"] + 
                      strikes_data.get(max_pain, {"CALL": {"OI": 0}, "PUT": {"OI": 0}})["PUT"]["OI"]) if max_pain else (current_price, 0, 0)
            e_point = max(significant_strikes, key=lambda x: abs(x[1])) if significant_strikes else (current_price + 2, 0, 0)

            wave_points = [a_point, b_point, c_point, d_point, e_point]
            wave_strikes = [point[0] for point in wave_points]
            wave_gamma = [point[1] for point in wave_points]
            wave_oi = [point[2] for point in wave_points]

            max_pain_divisions = [max_pain / s if s != 0 and max_pain is not None else 0 for s in wave_strikes]
            pressure = [oi / abs(s - max_pain) if max_pain is not None and s != max_pain and abs(s - max_pain) > 0 else 0 
                        for s, oi in zip(wave_strikes, wave_oi)]
            max_pressure = max(pressure) if pressure and max(pressure) > 0 else 1
            pressure_normalized = [p / max_pressure * 100 for p in pressure]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=strikes, y=call_gamma, name="CALL Gamma", marker_color="#32CD32", width=0.8, opacity=0.4))
            fig.add_trace(go.Bar(x=strikes, y=put_gamma, name="PUT Gamma", marker_color="#FF4500", width=0.8, opacity=0.4))
            fig.add_trace(go.Scatter(x=[current_price, current_price], y=[min(put_gamma) * 1.1, max(call_gamma) * 1.1], 
                                    mode="lines", line=dict(color="#FFFFFF", dash="dash", width=2), name="Current Price"))

            for i in range(len(wave_strikes) - 1):
                start_strike = wave_strikes[i]
                end_strike = wave_strikes[i + 1]
                start_gamma = wave_gamma[i]
                end_gamma = wave_gamma[i + 1]
                if start_strike is not None and end_strike is not None:
                    if (start_strike > current_price and end_strike < current_price) or \
                       (start_strike < current_price and end_strike > current_price):
                        line_color = "#FF4500" if start_strike > end_strike else "#32CD32"
                    else:
                        line_color = "#FFD700"
                    fig.add_trace(go.Scatter(x=[start_strike, end_strike], y=[start_gamma, end_gamma], 
                                            mode="lines", line=dict(color=line_color, width=1), showlegend=False))

            fig.add_trace(go.Scatter(x=wave_strikes, y=wave_gamma, mode="markers+text", name="Elliott Pulse",
                                    marker=dict(size=8, symbol="circle", color="#FFD700"),
                                    text=[f"${s:.2f}\n{letter}\n<span style='color:{'#FF4500' if p > 50 else '#32CD32'}'>{p:.0f}</span>" 
                                          for s, letter, p in zip(wave_strikes, ["A", "B", "C", "D", "E"], pressure_normalized)],
                                    textposition="top center",
                                    textfont=dict(size=12),
                                    customdata=[(s, g, o, mpd, p) for s, g, o, mpd, p in zip(wave_strikes, wave_gamma, wave_oi, max_pain_divisions, pressure_normalized)],
                                    hovertemplate="Strike: $%{customdata[0]:.2f}<br>Gamma: %{customdata[1]:.2f}<br>OI: %{customdata[2]:,d}<br>Max Pain / Strike: %{customdata[3]:.2f}<br>MM Pressure: %{customdata[4]:.0f"))
            fig.add_trace(go.Scatter(x=[max_pain], y=[max_pain_gamma], mode="markers+text", name="Max Pain",
                                    marker=dict(size=13, color="white", symbol="star"), text=[f"${max_pain:.2f}" if max_pain else "N/A"], 
                                    textposition="bottom center"))

            fig.update_layout(
                title=f"Elliott Pulse {ticker} (Exp: {selected_expiration})",
                xaxis_title="Strike Price",
                yaxis_title="Gamma Exposure",
                barmode="relative",
                template="plotly_dark",
                hovermode="x unified",
                height=600,
                legend=dict(yanchor="top", y=1.1, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0.5)"),
                plot_bgcolor="#000000",
                paper_bgcolor="#000000",
                font=dict(color="#FFFFFF"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)")
            )
            st.plotly_chart(fig, config={'staticPlot': False, 'displayModeBar': True}, use_container_width=True)
            elliott_df = pd.DataFrame({
                "Strike": strikes,
                "CALL_Gamma": call_gamma,
                "PUT_Gamma": put_gamma,
                "Net_Gamma": net_gamma,
                "Total_OI": total_oi,
                "Wave_Point": ["A" if s == wave_strikes[0] else "B" if s == wave_strikes[1] else "C" if s == wave_strikes[2] else "D" if s == wave_strikes[3] else "E" if s == wave_strikes[4] else "" for s in strikes],
                "Max_Pain_Division": [max_pain / s if s != 0 and max_pain is not None else 0 for s in strikes],
                "MM_Pressure": [oi / abs(s - max_pain) if max_pain is not None and s != max_pain and abs(s - max_pain) > 0 else 0 for s, oi in zip(strikes, total_oi)]
            })
            elliott_csv = elliott_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Elliott Pulse Data",
                data=elliott_csv,
                file_name=f"{ticker}_elliott_pulse_{selected_expiration}.csv",
                mime="text/csv",
                key="download_tab7"
            )
            st.markdown("---")
            st.markdown("*Developed by Ozy | © 2025*")

    # Tab 8: Crypto Insights
    with tab8:
        st.subheader("Crypto Insights")
        
        # Entrada del usuario
        ticker = st.text_input("Enter Crypto Ticker (e.g., BTC, ETH, XRP):", value="BTC", key="crypto_ticker_tab8").upper()
        selected_pair = f"{ticker}/USD"
        
        # Botón de actualización
        refresh_button = st.button("Refresh Orders", key="refresh_tab8")
        
        # Placeholder para actualización dinámica
        placeholder = st.empty()
        
        # Procesar datos al hacer clic o en la primera carga
        if refresh_button or "tab8_initialized" not in st.session_state:
            with st.spinner(f"Fetching data for {selected_pair}..."):
                try:
                    # Obtener datos de mercado de CoinGecko
                    market_data = fetch_coingecko_data(ticker)
                    if not market_data:
                        st.error(f"Failed to fetch market data for {ticker} from CoinGecko.")
                        logger.error(f"No market data returned for {ticker} from CoinGecko")
                    else:
                        # Obtener libro de órdenes de Kraken
                        logger.info(f"Attempting to fetch order book for {selected_pair}")
                        bids, asks, current_price = fetch_order_book(ticker, depth=500)
                        if bids.empty or asks.empty:
                            st.error(f"Failed to fetch order book for {selected_pair}. Verify the ticker or check Kraken API status.")
                            logger.error(f"Order book fetch failed: bids={len(bids)}, asks={len(asks)} for {selected_pair}")
                        else:
                            # Mostrar datos en el placeholder
                            with placeholder.container():
                                st.markdown(f"### {ticker} USD ({ticker}USD)")
                                st.write(f"**Price**: ${market_data['price']:,.2f}")
                                st.write(f"**Change (24h)**: {market_data['change_value']:,.2f} ({market_data['change_percent']:.2f}%)")
                                st.write(f"**Volume (24h)**: {market_data['volume']:,.0f}")
                                st.write(f"**Market Cap**: ${market_data['market_cap']:,.0f}")
                                
                                # Generar y mostrar gráfico de burbujas
                                fig, order_metrics = plot_order_book_bubbles_with_max_pain(bids, asks, current_price, ticker, market_data['volatility'])
                                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_tab8_{ticker}_{int(time.time())}")
                                
                                # Mostrar métricas
                                pressure_color = "#32CD32" if order_metrics['net_pressure'] > 0 else "#FF4500" if order_metrics['net_pressure'] < 0 else "#FFFFFF"
                                st.write(f"**Net Pressure**: <span style='color:{pressure_color}'>{order_metrics['net_pressure']:,.0f}</span> ({order_metrics['trend']})", unsafe_allow_html=True)
                                st.write(f"**Volatility (Annualized)**: {order_metrics['volatility']:.2f}%")
                                st.write(f"**Projected Target**: ${order_metrics['target_price']:,.2f}")
                                st.write(f"**Support**: ${order_metrics['support']:.2f} | **Resistance**: ${order_metrics['resistance']:.2f}")
                                st.write("**Whale Accumulation Zones**: " + ", ".join([f"${zone:.2f}" for zone in order_metrics['whale_zones']]))
                                edge_color = "#32CD32" if order_metrics['edge_score'] > 50 else "#FF4500" if order_metrics['edge_score'] < 30 else "#FFD700"
                                st.write(f"**Trader's Edge Score**: <span style='color:{edge_color}'>{order_metrics['edge_score']:.1f}</span> (0-100)", unsafe_allow_html=True)
                            
                            # Marcar como inicializado
                            st.session_state["tab8_initialized"] = True
                except Exception as e:
                    st.error(f"Error processing data for {selected_pair}: {str(e)}")
                    logger.error(f"Tab 8 error: {str(e)}")
        
        # Pie de página
        st.markdown("---")
        st.markdown("*Developed by Ozy | © 2025*")


    # Tab 9: Earnings Calendar
    with tab9:
        today = datetime.now().date()
        days_to_next_sunday = (6 - today.weekday() + 7) % 7
        if days_to_next_sunday == 0:
            days_to_next_sunday = 0
        end_of_next_week = today + timedelta(days=days_to_next_sunday + 7)
        start_date = today
        end_date = end_of_next_week

        sort_by = st.selectbox("Sort By", ["Date", "Symbol", "Possible Movement", "EPS", "Revenue", "Time"], index=0)

        @st.cache_data
        def fetch_api_data(url, params, headers, source):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=5)
                response.raise_for_status()
                logger.info(f"{source} API call successful: {len(response.json())} items")
                return response.json()
            except Exception as e:
                logger.error(f"Error fetching {source} data: {e}")
                return []

        def get_earnings_calendar(start_date: datetime, end_date: datetime) -> List[Dict]:
            url = f"{FMP_BASE_URL}/earning_calendar"
            params = {
                "apikey": FMP_API_KEY,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d")
            }
            data = fetch_api_data(url, params, HEADERS_FMP, "FMP Earnings")
            if not data or not isinstance(data, list):
                logger.error(f"No earnings data from FMP: {data}")
                st.error(f"No earnings data received from FMP: {data}")
                return []
            top_stocks = get_top_traded_stocks()
            earnings = []
            for item in data:
                symbol = item.get("symbol", "")
                event_date = item.get("date", "")
                try:
                    event_date_obj = datetime.strptime(event_date, "%Y-%m-%d").date()
                    if not (start_date <= event_date_obj <= end_date):
                        continue
                except ValueError:
                    continue
                eps_est = item.get("epsEstimated")
                revenue_est = item.get("revenueEstimated")
                time = item.get("time", "N/A").lower()
                try:
                    revenue_est = float(revenue_est) if revenue_est is not None else 0
                except (ValueError, TypeError):
                    revenue_est = 0
                if eps_est is None or revenue_est < 1_000_000:
                    continue
                
                if time == "bmo":
                    time_display = "Pre-Market"
                    time_factor = 1.0
                    time_sort_value = 0
                elif time == "amc":
                    time_display = "After-Market"
                    time_factor = 1.3
                    time_sort_value = 1
                else:
                    time_display = "N/A"
                    time_factor = 0.7
                    time_sort_value = 2

                base_volatility = 6.5
                volatility_factor = 12 * (1 + math.tanh(abs(eps_est) - 1))
                eps_relevance = abs(eps_est) / (revenue_est / 1_000_000_000 + 0.1)
                eps_impact = abs(eps_est) * volatility_factor * eps_relevance
                size_adjustment = 1 / (math.log10(revenue_est / 1_000_000 + 1) / 4 if revenue_est > 0 else 1)
                market_sensitivity = 0.9 + 0.2 * (1 - math.exp(-abs(eps_est)))
                movement = (base_volatility + eps_impact) * size_adjustment * time_factor * market_sensitivity
                movement = max(5.0, min(40.0, movement))

                earnings.append({
                    "Date": event_date,
                    "Time": time_display,
                    "TimeSortValue": time_sort_value,
                    "Symbol": symbol,
                    "Details": f"EPS: {eps_est:.2f} | Rev: ${revenue_est / 1_000_000:,.1f}M",
                    "Logo": fetch_logo_url(symbol),
                    "Possible Movement (%)": round(movement, 2),
                    "EPS": eps_est,
                    "Revenue": revenue_est,
                    "IsTopStock": symbol in top_stocks
                })
            logger.info(f"Processed {len(earnings)} earnings events")
            if not earnings:
                logger.warning("No earnings events after filtering")
            
            if sort_by == "Possible Movement":
                earnings.sort(key=lambda x: x["Possible Movement (%)"], reverse=True)
            elif sort_by == "EPS":
                earnings.sort(key=lambda x: x["EPS"], reverse=True)
            elif sort_by == "Revenue":
                earnings.sort(key=lambda x: x["Revenue"], reverse=True)
            elif sort_by == "Symbol":
                earnings.sort(key=lambda x: x["Symbol"])
            elif sort_by == "Time":
                earnings.sort(key=lambda x: (x["TimeSortValue"], x["Date"]))
            else:
                earnings.sort(key=lambda x: x["Date"])
            return earnings

        with st.spinner(f"Fetching earnings from {start_date} to {end_date}..."):
            earnings_events = get_earnings_calendar(start_date, end_date)
            logger.info(f"Finished fetching: {len(earnings_events)} events retrieved")

        if earnings_events:
            grouped_events = group_by_day_of_week(earnings_events, start_date, end_date)
            earnings_html = """
            <style>
                .earnings-container {
                    width: 100%;
                    background: linear-gradient(135deg, #1E1E1E, #2A2A2A);
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
                    overflow-y: auto;
                    max-height: 600px;
                }
                .date-section {
                    margin-bottom: 20px;
                }
                .date-header {
                    background-color: #2D2D2D;
                    color: #32CD32;
                    padding: 10px;
                    font-size: 18px;
                    font-weight: 600;
                    text-align: center;
                    border-radius: 5px;
                    cursor: default;
                }
                .cards-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    padding: 10px;
                }
                .earning-card {
                    width: 300px;
                    background: linear-gradient(145deg, #252525, #303030);
                    border: 1px solid #555555;
                    border-radius: 10px;
                    padding: 15px;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                .earning-card.top-stock {
                    border: 2px solid #FFD700;
                    background: linear-gradient(145deg, #303030, #404040);
                }
                .earning-card:hover {
                    transform: translateY(-5px) scale(1.05);
                    box-shadow: 0 10px 25px rgba(50, 205, 50, 0.2);
                }
                .earning-card img {
                    width: 80px;
                    height: 80px;
                    object-fit: contain;
                    border-radius: 50%;
                    margin: 0 auto;
                    display: block;
                }
                .earning-card .info {
                    color: #FFFFFF;
                    font-size: 14px;
                    text-align: center;
                    margin-top: 10px;
                }
                .earning-card .info .symbol {
                    font-size: 18px;
                    font-weight: 600;
                    color: #32CD32;
                }
                .earning-card .info .highlight {
                    color: #FFD700;
                }
                .tooltip {
                    visibility: hidden;
                    width: 250px;
                    background: #2D2D2D;
                    color: #FFFFFF;
                    text-align: left;
                    border-radius: 6px;
                    padding: 10px;
                    position: absolute;
                    z-index: 1;
                    top: 100%;
                    left: 50%;
                    margin-left: -125px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
                    font-size: 12px;
                }
                .earning-card:hover .tooltip {
                    visibility: visible;
                }
            </style>
            <div class="earnings-container">
            """

            if not grouped_events:
                earnings_html += "<p style='color: #FFFFFF; text-align: center;'>No events to display for the selected period.</p>"
            else:
                for day, events in grouped_events.items():
                    earnings_html += f"""
                    <div class="date-section">
                        <div class="date-header">{day}</div>
                        <div class="cards-container">
                    """
                    for event in events:
                        class_name = "earning-card top-stock" if event["IsTopStock"] else "earning-card"
                        logo = f'<img src="{event.get("Logo", "")}" alt="{event.get("Symbol", "N/A")}">'
                        tooltip = f"""
                            <div class="tooltip">
                                <b>Date:</b> {event["Date"]}<br>
                                <b>Time:</b> {event["Time"]}<br>
                                <b>Symbol:</b> {event["Symbol"]}<br>
                                <b>EPS:</b> {event["EPS"]:.2f}<br>
                                <b>Revenue:</b> ${event["Revenue"] / 1_000_000:,.1f}M<br>
                                <b>Possible Movement:</b> {event["Possible Movement (%)"]}%
                            </div>
                        """
                        earnings_html += f"""
                            <div class="{class_name}">
                                {logo}
                                <div class="info">
                                    <div class="symbol">{event["Symbol"]}</div>
                                    <div>{event["Time"]}</div>
                                    <div>{event["Details"]}</div>
                                    <div class="highlight">Move: {event["Possible Movement (%)"]}%</div>
                                    {tooltip}
                                </div>
                            </div>
                        """
                    earnings_html += """
                        </div>
                    </div>
                    """

            earnings_html += """
            </div>
            """

            components.html(earnings_html, height=600, scrolling=True)

            earnings_csv = pd.DataFrame(earnings_events).drop(columns=["Logo", "EPS", "Revenue", "TimeSortValue", "IsTopStock"], errors="ignore").to_csv(index=False)
            st.download_button(
                label="📥 Download Earnings Calendar",
                data=earnings_csv,
                file_name=f"earnings_calendar_{start_date}_to_{end_date}.csv",
                mime="text/csv",
                key="download_earnings_tab9"
            )
        else:
            st.info(f"No earnings events found from {start_date} to {end_date}. Check logs for details.")
            logger.warning("No earnings events retrieved or processed")
        
        st.markdown("---")
        st.markdown("*Developed by Ozy | © 2025*")

    # Tab 10: Psychological Edge
    with tab10:
        trading_points = [
            ("Sigue los Order Blocks del MM / Spot order blocks of the MM", 
             "Identifica zonas de order blocks (acumulación o distribución del MM) en niveles de liquidez clave. Opera con su flujo, no contra él. Camina antes de la sesión para alinear tu mente con su juego. / Spot order blocks (MM accumulation or distribution) at key liquidity zones. Trade with their flow, not against it. Walk before the session to sync your mind with their game.", 
             "#2A2A3D"),
            ("Domina el Riesgo Institucional / Master Institutional Risk", 
             "Limita el riesgo al 1-2% por posición, ajustado por VaR (Value at Risk) si operas portafolios grandes. No entres si tu estado emocional compromete el análisis de datos duros. / Cap risk at 1-2% per position, adjusted by VaR for large portfolios. Don’t trade if your emotional state clouds hard data analysis.", 
             "#1A2E2A"),
            ("Analiza la Exposición al Gamma / Analyze Gamma Exposure", 
             "Monitorea el gamma de opciones para prever cambios bruscos en el delta. Anticipa trampas del MM cerca de vencimientos (gamma squeezes) y reflexiona: ¿dónde están cazando stops? / Track options gamma to predict sharp delta shifts. Anticipate MM traps near expirations (gamma squeezes) and ask: Where are they hunting stops?", 
             "#3D2A2A"),
            ("Stop-Loss Basado en Liquidez / Liquidity-Based Stop-Loss", 
             "Coloca stops en niveles de liquidez del MM (debajo de soportes o encima de resistencias), no en zonas aleatorias. Corta rápido: en trading institucional, preservar capital es prioridad. Opera de pie para máxima alerta. / Set stops at MM liquidity levels (below support or above resistance), not random zones. Cut fast: in institutional trading, capital preservation is king. Trade standing for peak alertness.", 
             "#2A2A3D"),
            ("Confirma con Flujo Institucional / Confirm with Institutional Flow", 
             "Busca order flow en order blocks con picos de volumen y gamma. Usa herramientas como footprint charts o monitores de bloques si tienes acceso. Pregúntate: ¿qué están acumulando o descargando? / Look for order flow in order blocks with volume and gamma spikes. Use footprint charts or block monitors if available. Ask: What are they accumulating or unloading?", 
             "#1E2A3D"),
            ("Controla la Psicología bajo Presión / Control Psychology Under Pressure", 
             "Evita reaccionar a volatilidad intradía o ruido de mercado. La disciplina del MM es tu modelo: no te sobreapalances ni cedas a la euforia. Muévete cada hora para no quemarte. / Don’t react to intraday volatility or market noise. MM discipline is your model: no over-leverage or euphoria. Move hourly to avoid burnout.", 
             "#2E2A1E"),
            ("Simula con Datos Reales / Simulate with Real Data", 
             "Usa simulación con datos históricos de order flow y gamma para replicar movimientos del MM. Analiza cómo tu toma de decisiones resiste bajo presión institucional. / Simulate with historical order flow and gamma data to mirror MM moves. Test how your decision-making holds under institutional pressure.", 
             "#1A2E2A"),
            ("Sigue el Volumen Institucional / Track Institutional Volume", 
             "Alto volumen en niveles clave con cambios en gamma confirma compromiso del MM. Opera con ellos cuando el smart money entra; usa volume profile para precisión. Camina para mantener claridad. / High volume at key levels with gamma shifts confirms MM commitment. Trade with them when smart money steps in; use volume profile for precision. Walk to stay clear-headed.", 
             "#2A2A3D"),
            ("Registra Cada Movimiento / Log Every Move", 
             "Documenta trades con niveles de order blocks, gamma, volumen y resultados. Incluye tu estado mental para ajustar sesgos. Los institucionales viven de datos, no de intuición. / Log trades with order blocks, gamma, volume, and outcomes. Add your mental state to tweak biases. Institutionals live on data, not gut feel.", 
             "#1E2A3D"),
            ("Sal en Zonas de Liquidez / Exit at Liquidity Zones", 
             "Toma ganancias en resistencias del MM o cuando el gamma indique un reversal. Usa liquidity grabs a tu favor: el MM a menudo empuja precios para atrapar a los débiles. / Take profits at MM resistance or when gamma signals a reversal. Leverage liquidity grabs: MM often pushes prices to trap the weak.", 
             "#3D2A2A"),
            ("Filtra el Ruido del Retail / Filter Retail Noise", 
             "Ignora hype de redes sociales o noticias sin respaldo en order flow. Los institucionales confían en el DOM (Depth of Market) y el precio, no en titulares. / Ignore retail hype on social media or news without order flow backing. Institutionals trust DOM (Depth of Market) and price, not headlines.", 
             "#2E2A1E"),
            ("Ajusta por Volatilidad / Adjust for Volatility", 
             "En días de alta volatilidad (vencimientos, eventos macro), reduce tamaño de posición. En días tranquilos, busca order blocks profundos para entradas sólidas. Adapta como el MM. / On high-volatility days (expirations, macro events), shrink position size. On quiet days, target deep order blocks for strong entries. Adapt like the MM.", 
             "#1A2E2A"),
            ("Explota las Ineficiencias / Exploit Inefficiencies", 
             "Busca desbalances entre order flow y gamma (ej. short squeezes o stop runs). Los institucionales ganan donde el retail pierde: opera con ventaja, no con esperanza. / Hunt imbalances between order flow and gamma (e.g., short squeezes or stop runs). Institutionals win where retail loses: trade with edge, not hope.", 
             "#2A2A3D"),
            ("Piensa como el MM / Think Like the MM", 
             "Pregúntate: ¿dónde colocan liquidez para atraer volumen? Usa su lógica (atrapar stops, forzar reversals) para anticipar y alinearte. Reflexiona entre sesiones. / Ask: Where do they place liquidity to draw volume? Use their logic (trap stops, force reversals) to anticipate and align. Reflect between sessions.", 
             "#1E2A3D"),
            ("Monitorea el Gummy Data Bubbles® y la Volatilidad Implícita / Monitor Gummy Data Bubbles® and Implied Volatility", 
             "Analiza el skew de opciones (asimetría en volatilidad implícita) para detectar sesgos del MM hacia alzas o bajas. Un skew pronunciado puede señalar liquidity grabs o stop hunts. Usa esta data para afinar entradas. / Track options skew (implied volatility asymmetry) to spot MM bias toward upside or downside. Sharp skew can signal liquidity grabs or stop hunts. Use it to fine-tune entries.", 
             "#3D2A2A"),
            ("Aprovecha el Dark Pool Flow / Leverage Dark Pool Flow", 
             "Si tienes acceso Monitores de Ozytarget (volumen institucional oculto), busca confirmación de order blocks. El MM usa estos flujos para mover mercados sin alertar al retail. Alinea tus trades con este smart money. / If you have dark pool data (hidden institutional volume), confirm order blocks. MM uses these flows to move markets without tipping off retail. Align trades with this smart money.", 
             "#2E2A1E"),
            ("Juega el Juego del Spoofing Legal / Play the Legal Spoofing Game", 
             "Detecta patrones de spoofing en el DOM o tape. No luches contra ellos; úsalos para entrar cuando el precio revierta tras el flush de liquidez. Requiere velocidad y precisión. / Spot spoofing patterns (MM fake orders to mislead) in the DOM or tape. Don’t fight them; use them to enter when price reverses after the liquidity flush. Demands speed and precision.", 
             "#1A2E2A")
        ]

        for i, (title, content, color) in enumerate(trading_points, 1):
            with st.expander(f"{i}. {title}"):
                st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; border-radius: 5px; color: #FFFFFF;'>
                    {content}
                </div>
                """, unsafe_allow_html=True)

        st.subheader("Inversión a largo plazo para institucionales / Long-Term Investing for Institutionals")
        
        investing_points = [
            ("Horizontes por Edad y Objetivo / Horizons by Age and Goal", 
             "Invierte según el ciclo: 5-10 años para capital activo, 15-20 para crecimiento sostenido, 30+ para legado o fondos soberanos. Elige activos con fundamentales sólidos, no especulación. / Invest by cycle: 5-10 years for active capital, 15-20 for sustained growth, 30+ for legacy or sovereign funds. Pick assets with solid fundamentals, not speculation.", 
             "#2A2A3D"),
            ("Diversifica con Precisión / Diversify with Precision", 
             "Reparte entre clases de activos (acciones, bonos, materias primas) según correlaciones y riesgo ajustado (Sharpe ratio). Compra más en caídas si el order flow institucional lo respalda. / Spread across asset classes (stocks, bonds, commodities) by correlations and risk-adjusted return (Sharpe ratio). Buy more on dips if institutional order flow supports it.", 
             "#1A2E2A"),
            ("Foco en Valor Fundamental / Focus on Fundamental Value", 
             "Selecciona empresas con flujo de caja robusto, baja deuda y ventaja estructural. Ignora ruido de corto plazo: los institucionales miran décadas, no días. / Choose firms with strong cash flow, low debt, and structural edge. Ignore short-term noise: institutionals eye decades, not days.", 
             "#3D2A2A"),
            ("Reinversión Estratégica / Strategic Reinvestment", 
             "Usa ganancias para aumentar exposición en activos con smart money detrás. Aprovecha caídas para acumular: el MM exagera el miedo para comprar barato. / Reinvest profits to boost exposure to smart money-backed assets. Capitalize on dips: MM overplays fear to buy low.", 
             "#1E2A3D"),
            ("Paciencia Institucional / Institutional Patience", 
             "Mantén posiciones a través de ciclos económicos: el crecimiento global supera las crisis (dato histórico). Compra más en rojo con análisis de order flow, no pánico. / Hold through economic cycles: global growth outlasts crises (historical fact). Buy more in the red with order flow analysis, not panic.", 
             "#2E2A1E"),
            ("Visión Macro Avanzada / Advanced Macro Vision", 
             "Invierte en sectores con respaldo institucional (tecnología, infraestructura, energía limpia) según datos macro (PIB, tasas, demografía). Ajusta con calma, no por modas. / Invest in institutionally backed sectors (tech, infrastructure, clean energy) based on macro data (GDP, rates, demographics). Adjust calmly, not on trends.", 
             "#2A2A3D"),
            ("Contrario con Datos / Contrarian with Data", 
             "Aumenta posiciones en caídas si el volume profile y fundamentales lo justifican. El MM usa el pánico retail para acumular; tú haz lo mismo con ventaja. / Build positions on dips if volume profile and fundamentals hold. MM uses retail panic to accumulate; you do the same with an edge.", 
             "#1A2E2A"),
            ("Gestión Activa Pasiva / Active-Passive Management", 
             "Combina tenencia pasiva (ETFs, índices) con compras activas en order blocks de largo plazo. Rebalancea según smart money, no emociones. / Blend passive holding (ETFs, indexes) with active buys at long-term order blocks. Rebalance by smart money, not emotion.", 
             "#3D2A2A"),
            ("Incorpora Derivados para Cobertura / Use Derivatives for Hedging", 
             "Usa futuros o opciones para cubrir portafolios contra caídas sin vender activos clave. Los institucionales protegen ganancias sin sacrificar exposición a largo plazo. Ajusta según gamma y volatility index (VIX). / Use futures or options to hedge portfolios against dips without selling core assets. Institutionals shield gains without losing long-term exposure. Adjust by gamma and volatility index (VIX).", 
             "#1E2A3D"),
            ("Explota Ciclos de Rebalancing / Exploit Rebalancing Cycles", 
             "Aprovecha ventanas de rebalancing institucional (fin de mes, trimestre) cuando el MM ajusta posiciones. Compra en caídas inducidas por estos flujos; el volume profile te dirá dónde entran. / Exploit institutional rebalancing windows (month-end, quarter-end) when MM adjusts positions. Buy dips driven by these flows; volume profile shows where they step in.", 
             "#2E2A1E")
        ]

        for i, (title, content, color) in enumerate(investing_points, 1):
            with st.expander(f"{i}. {title}"):
                st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; border-radius: 5px; color: #FFFFFF;'>
                    {content}
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("*Developed by Ozy | © 2025*")

    # Tab 11: Projection
    with tab11:
        # Estilo CSS
        st.markdown("""
            <style>
            .main-title { 
                font-size: 28px; 
                font-weight: 600; 
                color: #FFFFFF; 
                text-align: center; 
                margin-bottom: 20px; 
                text-shadow: 0 0 5px rgba(50, 205, 50, 0.5); 
            }
            .section-header { 
                font-size: 20px; 
                font-weight: 500; 
                color: #32CD32; 
                margin-top: 20px; 
                border-bottom: 1px solid #32CD32; 
                padding-bottom: 5px; 
            }
            .metric-label { 
                font-size: 16px; 
                color: #FFFFFF; 
                font-family: 'Arial', sans-serif; 
            }
            .metric-value { 
                font-size: 18px; 
                font-weight: 600; 
                color: #FFD700; 
            }
            .tooltip { 
                position: relative; 
                display: inline-block; 
                cursor: help; 
                color: #32CD32; 
                margin-left: 5px; 
            }
            .tooltip .tooltiptext { 
                visibility: hidden; 
                width: 200px; 
                background-color: #2D2D2D; 
                color: #FFFFFF; 
                text-align: center; 
                border-radius: 5px; 
                padding: 5px; 
                position: absolute; 
                z-index: 1; 
                bottom: 125%; 
                left: 50%; 
                margin-left: -100px; 
                font-size: 12px; 
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5); 
            }
            .tooltip:hover .tooltiptext { 
                visibility: visible; 
            }
            </style>
        """, unsafe_allow_html=True)

        ticker = st.text_input("Ticker Symbol (e.g., TSLA, NVDA)", "NVDA", key="institutional_ticker").upper()

        with st.spinner(f"Fetching real-time data for {ticker}..."):
            try:
                # Funciones internas optimizadas
                @st.cache_data(ttl=60)
                def get_intraday_data(ticker: str, interval="1min", limit=5) -> Tuple[List[float], List[int]]:
                    url = f"{TRADIER_BASE_URL}/markets/history"
                    start_time = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    end_time = datetime.now().strftime("%Y-%m-%d")
                    params = {"symbol": ticker, "interval": interval, "start": start_time, "end": end_time}
                    data = fetch_api_data(url, params, HEADERS_TRADIER, "Tradier Intraday")
                    if data and "history" in data and "day" in data["history"]:
                        prices = [float(day["close"]) for day in data["history"]["day"][-limit:]]
                        volumes = [int(day["volume"]) for day in data["history"]["day"][-limit:]]
                        return prices, volumes
                    return [get_current_price(ticker)] * limit, [0] * limit

                @st.cache_data(ttl=60)
                def get_vix() -> float:
                    url = f"{FMP_BASE_URL}/quote/^VIX"
                    params = {"apikey": FMP_API_KEY}
                    data = fetch_api_data(url, params, HEADERS_FMP, "VIX")
                    return float(data[0]["price"]) if data and isinstance(data, list) and "price" in data[0] else 20.0

                @st.cache_data(ttl=300)
                def get_news_sentiment(ticker: str) -> float:
                    keywords = [ticker]
                    news = fetch_google_news(keywords)
                    if not news:
                        return 0.5
                    sentiment = sum(1 if "up" in article["title"].lower() else -1 if "down" in article["title"].lower() else 0 for article in news)
                    return max(0, min(1, 0.5 + sentiment / (len(news) * 2)))

                def calculate_probability_cone(current_price: float, iv: float, days: List[int]) -> Dict:
                    cone = {}
                    for day in days:
                        sigma = iv * current_price * (day / 365) ** 0.5
                        cone[day] = {
                            "68_lower": current_price - sigma,
                            "68_upper": current_price + sigma,
                            "95_lower": current_price - 2 * sigma,
                            "95_upper": current_price + 2 * sigma
                        }
                    return cone

                # Obtener datos en tiempo real
                current_price = get_current_price(ticker)
                if current_price == 0.0:
                    st.error(f"Could not retrieve real-time price for {ticker}.")
                    st.stop()

                prices_1m, volumes_1m = get_intraday_data(ticker)

                # Perfil de la empresa (FMP)
                url_profile = f"{FMP_BASE_URL}/profile/{ticker}"
                params_profile = {"apikey": FMP_API_KEY}
                profile_data = fetch_api_data(url_profile, params_profile, HEADERS_FMP, "FMP Profile")
                if not profile_data or not isinstance(profile_data, list) or len(profile_data) == 0:
                    st.error(f"No fundamental data found for {ticker}.")
                    st.stop()
                profile = profile_data[0]
                market_cap = profile.get("mktCap", 1_000_000_000)
                pe_ratio = profile.get("pe", 20)
                pb_ratio = profile.get("pb", 3)
                debt_to_equity = profile.get("debtToEquity", 1.0)
                roe = profile.get("roe", 0.1)
                profit_margin = profile.get("profitMargin", 0.05)
                beta = profile.get("beta", 1.0)
                sector = profile.get("sector", "Unknown")

                # Datos históricos (1 año)
                prices, volumes = get_historical_prices_combined(ticker, limit=252)
                if not prices or len(prices) < 20:
                    prices = [current_price] * 20
                    volumes = [1_000_000] * 20
                returns = np.diff(prices) / prices[:-1]
                vol_historical = np.std(returns) * np.sqrt(252)

                # Datos de opciones
                expiration_dates = get_expiration_dates(ticker)
                iv = get_implied_volatility(ticker) or 0.3
                gamma_exposure = 0
                skew = 0
                vmi = 0
                oi_by_strike = {}
                oi_total = 0
                gamma_wall = 0
                if expiration_dates:
                    options_data = get_options_data(ticker, expiration_dates[0])
                    if options_data and isinstance(options_data, list):
                        gamma_total = sum(float(opt.get("greeks", {}).get("gamma", 0)) * int(opt.get("open_interest", 0)) 
                                          for opt in options_data if "greeks" in opt)
                        oi_total = sum(int(opt.get("open_interest", 0)) for opt in options_data)
                        gamma_exposure = gamma_total * current_price / max(1, oi_total) if oi_total > 0 else 0
                        
                        calls_iv_list = [float(opt["greeks"]["smv_vol"]) for opt in options_data 
                                         if opt.get("option_type", "").lower() == "call" and "greeks" in opt and opt["greeks"].get("smv_vol", 0) > 0]
                        puts_iv_list = [float(opt["greeks"]["smv_vol"]) for opt in options_data 
                                        if opt.get("option_type", "").lower() == "put" and "greeks" in opt and opt["greeks"].get("smv_vol", 0) > 0]
                        calls_iv = np.mean(calls_iv_list) if calls_iv_list else iv
                        puts_iv = np.mean(puts_iv_list) if puts_iv_list else iv
                        iv = np.mean(calls_iv_list + puts_iv_list) if calls_iv_list or puts_iv_list else iv
                        skew = (calls_iv - puts_iv) / iv if calls_iv_list and puts_iv_list and iv != 0 else 0
                        vmi = (skew * oi_total / max(1, vol_historical * 100)) if oi_total > 0 else 0
                        
                        gamma_by_strike = {strike: sum(float(opt.get("greeks", {}).get("gamma", 0)) * int(opt.get("open_interest", 0)) 
                                                      for opt in options_data if float(opt["strike"]) == strike and "greeks" in opt) 
                                          for strike in set(float(opt["strike"]) for opt in options_data)}
                        gamma_wall = max(gamma_by_strike.items(), key=lambda x: abs(x[1]), default=(current_price, 0))[0] if gamma_by_strike else current_price
                        
                        strikes = [float(opt["strike"]) for opt in options_data]
                        oi_by_strike = {strike: sum(int(opt.get("open_interest", 0)) for opt in options_data if float(opt["strike"]) == strike) 
                                        for strike in set(strikes)}
                    else:
                        st.warning("No valid options data returned from Tradier.")
                else:
                    st.warning("No expiration dates available for options data.")

                # Cálculos institucionales avanzados
                volume_delta = sum([v if p > prices_1m[i-1] else -v for i, (p, v) in enumerate(zip(prices_1m[1:], volumes_1m[1:]))])
                oi_delta = oi_total
                gamma_weighted = gamma_exposure * sum(1/abs(s - current_price) for s in oi_by_strike.keys() if oi_by_strike[s] > 0) if oi_by_strike else 0
                vix = get_vix()
                ifm = min(100, max(0, (volume_delta * oi_delta * gamma_weighted) / (vix + iv * 100))) if (vix + iv * 100) != 0 else 0

                oi_below = sum(oi for s, oi in oi_by_strike.items() if s < current_price)
                oi_above = sum(oi for s, oi in oi_by_strike.items() if s > current_price)
                vwap_daily = sum(p * v for p, v in zip(prices[-20:], volumes[-20:])) / sum(volumes[-20:]) if sum(volumes[-20:]) > 0 else current_price
                lti = (oi_below / oi_above if oi_above > 0 else 1.0) * (current_price - vwap_daily) / (iv * beta) if (iv * beta) != 0 else 0

                event_factor = 1.5 if fetch_api_data(f"{FMP_BASE_URL}/economic-calendar", {"apikey": FMP_API_KEY, "from": datetime.now().strftime("%Y-%m-%d"), "to": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")}, HEADERS_FMP, "Events") else 1.0
                earnings_data = get_earnings_calendar(datetime.now().date(), datetime.now().date() + timedelta(days=5))
                earnings_factor = 1.8 if any(e.get("date") and datetime.strptime(e["date"], "%Y-%m-%d").date() <= (datetime.now().date() + timedelta(days=5)) for e in earnings_data) else 1.0
                skew_impact = skew + 0.1
                eaem = iv * current_price * (1 + abs(gamma_exposure)) * event_factor * skew_impact * earnings_factor * 0.15
                eaem_upper = current_price + eaem
                eaem_lower = max(0, current_price - eaem)

                eaem_ratio = (current_price - eaem_lower) / (eaem_upper - eaem_lower) if (eaem_upper - eaem_lower) != 0 else 0.5
                sentiment_score = get_news_sentiment(ticker)
                rtes = (ifm * 0.4 + lti * 0.3 + eaem_ratio * 20 + sentiment_score * 10)

                # Cálculo OIPI original
                volume_avg = np.mean(volumes[-20:])
                volume_spike = max(volumes[-5:]) / volume_avg if volume_avg > 0 else 1.0
                price_change = (current_price - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
                momentum_score = min(30, 8 * volume_spike + 8 * abs(price_change) * 100 + 14 * (gamma_exposure / max(1, abs(gamma_exposure))))
                momentum_text = "Strong institutional momentum" if momentum_score > 20 else "Moderate activity" if momentum_score > 10 else "Low momentum"

                sharpe_ratio = (np.mean(returns) * 252 - RISK_FREE_RATE) / vol_historical if vol_historical > 0 else 1.0
                growth_factor = roe * profit_margin
                debt_factor = 1 / (1 + debt_to_equity) if debt_to_equity > 0 else 1
                health_score = min(30, 10 * sharpe_ratio + 10 * growth_factor + 10 * debt_factor)
                health_text = "Robust fundamentals" if health_score > 20 else "Stable with risks" if health_score > 10 else "Weak fundamentals"

                sector_pe_avg = {"Technology": 30, "Financial Services": 15, "Healthcare": 25, "Consumer Cyclical": 20}.get(sector, 20)
                pe_factor = min(1.5, sector_pe_avg / pe_ratio) if pe_ratio > 0 else 1.0
                pb_factor = min(1.5, 3 / pb_ratio) if pb_ratio > 0 else 1.0
                cash_flow_est = max(market_cap * profit_margin, 1_000_000) * (1 + roe * 0.5)
                discount_rate = max(RISK_FREE_RATE + beta * 0.06, 0.01)
                fair_value = (cash_flow_est / discount_rate) / 1_000_000_000
                fair_value = max(fair_value, current_price * 0.5)
                fair_value_text = "Below Fair Value" if current_price < fair_value else "At Fair Value" if abs(current_price - fair_value) < current_price * 0.05 else "Above Fair Value"
                valuation_score = min(20, 8 * pe_factor + 8 * pb_factor + 4 * (current_price / fair_value if fair_value > 0 else 1))
                valuation_text = "Undervalued" if valuation_score > 15 else "Fair" if valuation_score > 10 else "Overvalued"

                risk_score = min(10, max(0, 10 - (iv * beta + abs(skew) * 5)))
                risk_text = "Low risk" if risk_score > 7 else "Moderate risk" if risk_score > 3 else "High risk"

                expected_move = iv * current_price * (1 + abs(gamma_exposure / max(1, gamma_exposure))) * 0.15
                move_score = min(10, expected_move / current_price * 100)
                upper_target = current_price + expected_move
                lower_target = current_price - expected_move

                oipi_score = momentum_score + health_score + valuation_score + risk_score + move_score
                oipi_score = max(0, min(100, oipi_score))
                oipi_recommendation = "Strong Buy" if oipi_score > 80 else "Buy" if oipi_score > 60 else "Hold" if oipi_score > 40 else "Sell"

                # Métricas adicionales (corto, mediano, largo plazo)
                short_term_risk = iv * current_price * (1 / 12)**0.5
                short_term_lower = current_price - short_term_risk
                short_term_upper = current_price + short_term_risk

                mid_term_risk = iv * current_price * (6 / 12)**0.5
                mid_term_lower = current_price - mid_term_risk
                mid_term_upper = current_price + mid_term_risk

                long_term_risk = iv * current_price * beta
                long_term_lower = current_price - long_term_risk
                long_term_upper = current_price + long_term_risk

                support = min([s for s in oi_by_strike.keys() if s < current_price], default=current_price - short_term_risk * 1.5) if oi_by_strike else current_price - short_term_risk * 1.5
                resistance = max([s for s in oi_by_strike.keys() if s > current_price], default=current_price + short_term_risk * 1.5) if oi_by_strike else current_price + short_term_risk * 1.5
                safe_zone_lower = max(support, fair_value * 0.9)
                safe_zone_upper = min(resistance, fair_value * 1.1)

                # Promedio dinámico por sector
                sector_benchmarks = {
                    "Technology": [0.8, 0.6, 0.7, 0.7, 0.8],
                    "Financial Services": [0.6, 0.7, 0.6, 0.8, 0.5],
                    "Healthcare": [0.7, 0.65, 0.75, 0.6, 0.7],
                    "Consumer Cyclical": [0.65, 0.6, 0.65, 0.7, 0.75]
                }
                sector_avg = sector_benchmarks.get(sector, [0.7, 0.6, 0.65, 0.8, 0.75])

                # Visualización
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown('<div class="section-header">Market </div>', unsafe_allow_html=True)
                    fig_heatmap = go.Figure()
                    price_points = [long_term_lower, mid_term_lower, short_term_lower, safe_zone_lower, current_price, safe_zone_upper, short_term_upper, mid_term_upper, long_term_upper]
                    y_values = [1] * len(price_points)
                    colors = ["#FF4500", "#FF8C00", "#FFD700", "#32CD32", "#FFFFFF", "#32CD32", "#FFD700", "#FF8C00", "#FF4500"]
                    fig_heatmap.add_trace(go.Scatter(x=price_points, y=y_values, mode="markers+text", text=[f"${p:.2f}" for p in price_points], textposition="top center", marker=dict(size=12, color=colors), hoverinfo="x+text"))
                    fig_heatmap.add_shape(type="rect", x0=safe_zone_lower, y0=0.8, x1=safe_zone_upper, y1=1.2, fillcolor="green", opacity=0.3, line_width=0)
                    for strike, oi in oi_by_strike.items():
                        if oi > oi_total * 0.05:
                            pressure = 1 if strike > current_price else -1
                            fig_heatmap.add_shape(type="line", x0=strike, y0=0.9, x1=strike, y1=1.1, line=dict(color="green" if pressure > 0 else "red", width=oi/oi_total*10, dash="dot"))
                    fig_heatmap.add_shape(type="line", x0=fair_value, y0=0, x1=fair_value, y1=2, line=dict(color="blue", dash="dash"))
                    fig_heatmap.add_annotation(x=fair_value, y=1.5, text=f"Fair: ${fair_value:.2f}", showarrow=False, font=dict(color="blue"))
                    fig_heatmap.add_shape(type="line", x0=current_price, y0=0, x1=current_price, y1=2, line=dict(color="white", width=2))
                    fig_heatmap.add_annotation(x=current_price, y=1.7, text=f"Now: ${current_price:.2f}", showarrow=False, font=dict(color="white"))
                    fig_heatmap.add_shape(type="line", x0=gamma_wall, y0=0, x1=gamma_wall, y1=2, line=dict(color="purple", width=1, dash="dot"))
                    fig_heatmap.add_annotation(x=gamma_wall, y=1.9, text=f"Gamma Wall: ${gamma_wall:.2f}", showarrow=False, font=dict(color="purple"))
                    fig_heatmap.update_layout(title="Order Flow & Liquidity", xaxis_title="Price", yaxis=dict(showgrid=False, showticklabels=False, range=[0, 2]), template="plotly_dark", height=300)
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    st.markdown('<div class="section-header">Probability Outlook</div>', unsafe_allow_html=True)
                    cone = calculate_probability_cone(current_price, iv, [1, 5, 30])
                    fig_cone = go.Figure()
                    for day in [1, 5, 30]:
                        fig_cone.add_trace(go.Scatter(x=[cone[day]["68_lower"], cone[day]["68_upper"]], y=[day, day], mode="lines", line=dict(color="#FFD700", width=1), name=f"{day}-Day 68%"))
                        fig_cone.add_trace(go.Scatter(x=[cone[day]["95_lower"], cone[day]["95_upper"]], y=[day, day], mode="lines", line=dict(color="#FF4500", width=1, dash="dash"), name=f"{day}-Day 95%"))
                    fig_cone.add_trace(go.Scatter(x=[current_price], y=[0], mode="markers", marker=dict(size=10, color="white"), name="Current Price"))
                    fig_cone.update_layout(title="Probability Cone", xaxis_title="Price Range", yaxis_title="Days Ahead", template="plotly_dark", height=300)
                    st.plotly_chart(fig_cone, use_container_width=True)

                    st.markdown('<div class="section-header">Performance </div>', unsafe_allow_html=True)
                    fig_radar = go.Figure()
                    categories = ["Momentum", "Health", "Valuation", "Risk", "VMI", "Momentum"]
                    scores = [momentum_score/30, health_score/30, valuation_score/20, risk_score/10, abs(vmi)*10, momentum_score/30]
                    texts = [momentum_text, health_text, valuation_text, risk_text, f"Vol Momentum: {vmi:.2f}", momentum_text]
                    absolutes = [f"{momentum_score:.1f}/30", f"{health_score:.1f}/30", f"{valuation_score:.1f}/20", f"{risk_score:.1f}/10", f"{abs(vmi):.2f}", f"{momentum_score:.1f}/30"]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=categories,
                        fill="toself",
                        name=ticker,
                        line=dict(color="#32CD32", width=2),
                        hovertemplate="%{theta}<br>Score: %{customdata[0]}<br>Rating: %{customdata[1]}",
                        customdata=list(zip(absolutes, texts)),
                        hoverlabel=dict(bgcolor="rgba(200, 200, 200, 0.8)", font_color="black")
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=sector_avg,
                        theta=categories,
                        fill="toself",
                        name=f"{sector} Avg",
                        line=dict(color="#FFD700", width=1),
                        opacity=0.5,
                        hovertemplate="%{theta}: %{r:.2f}<br>",
                        hoverlabel=dict(bgcolor="rgba(200, 200, 200, 0.8)", font_color="black")
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="white")),
                            angularaxis=dict(tickfont=dict(color="white"))
                        ),
                        showlegend=True,
                        title=f"Performance Radar | {ticker} ({oipi_score:.1f}/100)",
                        template="plotly_dark",
                        height=300,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    color_rtes = "#32CD32" if rtes > 60 else "#FFD700" if rtes > 40 else "#FF4500"
                    st.markdown(f'<span class="metric-label">Real-Time Edge Score:</span> <span class="metric-value" style="color:{color_rtes}">{rtes:.1f}/100</span> <span class="tooltip">ℹ️<span class="tooltiptext">Overall trading edge based on momentum, liquidity, and sentiment</span></span>', unsafe_allow_html=True)
                    rtes_recommendation = "Strong Buy" if rtes > 80 else "Buy" if rtes > 60 else "Hold" if rtes > 40 else "Sell"
                    st.markdown(f'<span class="metric-label">Recommendation:</span> <span class="metric-value">{rtes_recommendation}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">IFM (Momentum):</span> <span class="metric-value">{ifm:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Institutional Flow Momentum: Measures smart money activity</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">LTI (Trap Index):</span> <span class="metric-value">{lti:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Liquidity Trap Index: Detects potential MM traps</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">EAEM Range:</span> <span class="metric-value">${eaem_lower:.2f} - ${eaem_upper:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Event-Adjusted Expected Move: Price range considering events</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Gamma Wall:</span> <span class="metric-value">${gamma_wall:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Strike with highest gamma pressure, key support/resistance</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">VMI:</span> <span class="metric-value">{vmi:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Volatility Momentum Index: Combines skew and OI for trend strength</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Sentiment:</span> <span class="metric-value">{sentiment_score:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">News sentiment (0 bearish, 1 bullish)</span></span>', unsafe_allow_html=True)

                    color_oipi = "#32CD32" if oipi_score > 60 else "#FFD700" if oipi_score > 40 else "#FF4500"
                    st.markdown(f'<span class="metric-label">OIPI:</span> <span class="metric-value" style="color:{color_oipi}">{oipi_score:.1f}/100</span> <span class="tooltip">ℹ️<span class="tooltiptext">Overall Potential Index: Long-term value score</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Recommendation:</span> <span class="metric-value">{oipi_recommendation}</span>', unsafe_allow_html=True)
                    insight = (
                        "Buy Now: Elite opportunity" if oipi_score > 80 else
                        "Accumulate: Strong potential" if oipi_score > 60 else
                        "Hold: Evaluate risks" if oipi_score > 40 else
                        "Sell: Weak outlook"
                    )
                    st.markdown(f'<span class="metric-label">Insight:</span> <span class="metric-value">{insight}</span>', unsafe_allow_html=True)

                    st.markdown('<div class="section-header"> </div>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Current Price:</span> <span class="metric-value">${current_price:.2f}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Fair Value:</span> <span class="metric-value">${fair_value:.2f} - {fair_value_text}</span> <span class="tooltip">ℹ️<span class="tooltiptext">DCF-based intrinsic value</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Safe Zone:</span> <span class="metric-value">${safe_zone_lower:.2f} - ${safe_zone_upper:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Range between support and resistance</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">SHORT:</span> <span class="metric-value">${short_term_lower:.2f} - ${short_term_upper:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">1-month expected range</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">MEDIUM:</span> <span class="metric-value">${mid_term_lower:.2f} - ${mid_term_upper:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">6-month expected range</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">LONG:</span> <span class="metric-value">${long_term_lower:.2f} - ${long_term_upper:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Long-term range adjusted by beta</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Gamma:</span> <span class="metric-value">{gamma_exposure:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Gamma exposure: Sensitivity to price changes</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Options Skew:</span> <span class="metric-value">{skew:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Call vs Put IV difference: Market bias</span></span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="metric-label">Sharpe Ratio:</span> <span class="metric-value">{sharpe_ratio:.2f}</span> <span class="tooltip">ℹ️<span class="tooltiptext">Risk-adjusted return</span></span>', unsafe_allow_html=True)

                    # Descarga de datos
                    data = {
                        "Ticker": ticker, "RTES": rtes, "IFM": ifm, "LTI": lti, "EAEM_Lower": eaem_lower, "EAEM_Upper": eaem_upper,
                        "Gamma_Wall": gamma_wall, "VMI": vmi, "OIPI": oipi_score, "OIPI_Recommendation": oipi_recommendation,
                        "Momentum": momentum_text, "Health": health_text, "Valuation": valuation_text, "Risk": risk_text,
                        "Current_Price": current_price, "Fair_Value": fair_value, "Safe_Zone_Lower": safe_zone_lower,
                        "Safe_Zone_Upper": safe_zone_upper, "Short_Term_Lower": short_term_lower, "Short_Term_Upper": short_term_upper,
                        "Mid_Term_Lower": mid_term_lower, "Mid_Term_Upper": mid_term_upper, "Long_Term_Lower": long_term_lower,
                        "Long_Term_Upper": long_term_upper, "IV": iv, "Gamma_Exposure": gamma_exposure, "Options_Skew": skew,
                        "Sharpe_Ratio": sharpe_ratio, "Market_Cap": market_cap, "P/E": pe_ratio, "P/B": pb_ratio,
                        "Debt/Equity": debt_to_equity, "ROE": roe, "Profit_Margin": profit_margin
                    }
                    df = pd.DataFrame([data])
                    csv = df.to_csv(index=False)
                    st.download_button(label="📥 Download EdgeMaster Data", data=csv, file_name=f"{ticker}_edgemaster.csv", mime="text/csv", key="download_edge")

                st.markdown("---")
                st.markdown("*Developed by Ozy | © 2025*", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing {ticker}: {str(e)}")
                import traceback
                logger.error(f"Tab 11 Pro Dashboard error: {traceback.format_exc()}")
                st.markdown("---")
                st.markdown("*Developed by Ozy | © 2025*", unsafe_allow_html=True)

    # Tab 12: Performance Map
    with tab12:
        # Estilo visual alineado con tab11
        st.markdown("""
            <style>
            .tab12-container {
                background: linear-gradient(135deg, #0d1b2a, #1b263b);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6);
                margin-bottom: 25px;
                position: relative;
                overflow: hidden;
                animation: fadeIn 1s ease-in-out;
            }
            .tab12-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle, rgba(233, 69, 96, 0.1), transparent 70%);
                animation: pulse 10s infinite;
                z-index: 0;
            }
            @keyframes pulse {
                0% { transform: scale(1); opacity: 0.3; }
                50% { transform: scale(1.2); opacity: 0.5; }
                100% { transform: scale(1); opacity: 0.3; }
            }
            @keyframes fadeIn {
                0% { opacity: 0; }
                100% { opacity: 1; }
            }
            .main-title {
                font-size: 28px;
                font-weight: 600;
                color: #FFFFFF;
                text-align: center;
                margin-bottom: 20px;
                text-shadow: 0 0 5px rgba(50, 205, 50, 0.5);
                z-index: 1;
                position: relative;
            }
            .section-header {
                font-size: 20px;
                font-weight: 500;
                color: #32CD32;
                margin-top: 20px;
                border-bottom: 1px solid #32CD32;
                padding-bottom: 5px;
                z-index: 1;
                position: relative;
            }
            .tab12-table th {
                background-color: #1b263b;
                color: #e94560;
                font-weight: 700;
                padding: 14px;
                border: 2px solid #415a77;
                text-transform: uppercase;
                font-size: 14px;
            }
            .tab12-table td {
                background-color: #0d1b2a;
                color: #e0e1dd;
                padding: 12px;
                border: 2px solid #415a77;
                text-align: center;
                transition: background-color 0.3s ease;
            }
            .tab12-table td:hover {
                background-color: #1b263b;
            }
            .metric-label {
                font-size: 16px;
                color: #FFFFFF;
                font-family: 'Arial', sans-serif;
            }
            .metric-value {
                font-size: 18px;
                font-weight: 600;
                color: #FFD700;
            }
            .tab12-implications-card {
                background-color: #1b263b;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                color: #e0e1dd;
                font-size: 14px;
                line-height: 1.6;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                border-left: 4px solid #e94560;
                transition: transform 0.3s ease;
                z-index: 1;
                position: relative;
            }
            .tab12-implications-card:hover {
                transform: translateX(5px);
            }
            .tab12-download {
                display: block;
                width: 220px;
                margin: 25px auto;
                padding: 12px;
                background-color: #e94560;
                color: #ffffff;
                text-align: center;
                border-radius: 6px;
                text-decoration: none;
                font-weight: bold;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }
            .tab12-download:hover {
                background-color: #ff6b81;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="tab12-container">', unsafe_allow_html=True)
        
        sectors_bonds = {
            "Basic Materials": "XLB",
            "Consumer Cyclical": "XLY",
            "Financials": "XLF",
            "Defense (Aerospace)": "XAR",
            "Real Estate": "XLRE",
            "Utilities": "XLU",
            "Communication Services": "XLC",
            "Healthcare": "XLV",
            "Energy": "XLE",
            "Industrials": "XLI",
            "Technology": "XLK",
            "20+ Yr Treasury Bond": "TLT",
            "7-10 Yr Treasury Bond": "IEF",
            "1-3 Yr Treasury Bond": "SHY",
            "VIX (Volatility Index)": "^VIX",
            "Dollar Index (UUP)": "UUP",
            "S&P 500 (SPY)": "SPY"
        }

        def calculate_sector_prediction(row, vix_data, dollar_data, spy_data, macro_factors):
            period_weights = {"1 Hour": 0.25, "1 Day": 0.25, "1 Week": 0.2, "1 Month": 0.15, "1 Quarter": 0.1, "1 Year": 0.05}
            momentum = sum(row[p] * w for p, w in period_weights.items())
            spy_momentum = sum(spy_data[p] * w for p, w in period_weights.items())
            correlation = 0.5 if row["Sector/Bond"] in ["Technology", "Consumer Cyclical", "Financials"] else 0.3
            vix_factor = (vix_data["1 Hour"] + vix_data["1 Day"]) * 0.125
            dollar_factor = (dollar_data["1 Hour"] + dollar_data["1 Day"]) * (-0.075 if "Treasury" not in row["Sector/Bond"] else 0.05)
            macro_factor = (
                macro_factors["fed_rate"] * -0.15 + macro_factors["gdp"] * 0.05 +
                macro_factors["cpi"] * -0.1 + macro_factors["unemployment"] * -0.08
            )
            ipms_score = momentum + correlation * spy_momentum + vix_factor + dollar_factor + macro_factor
            direction = "Up" if ipms_score > 0.3 else "Down" if ipms_score < -0.3 else "Neutral"
            magnitude = min(abs(ipms_score) * 1.5, 4.0)
            return direction, magnitude, ipms_score

        def calculate_market_prediction(df, macro_factors):
            spy_data = df[df["Sector/Bond"] == "S&P 500 (SPY)"].iloc[0]
            vix_data = df[df["Sector/Bond"] == "VIX (Volatility Index)"].iloc[0]
            dollar_data = df[df["Sector/Bond"] == "Dollar Index (UUP)"].iloc[0]
            sectors = df[~df["Sector/Bond"].str.contains("Treasury|VIX|Dollar|SPY")]
            period_weights = {"1 Hour": 0.25, "1 Day": 0.25, "1 Week": 0.2, "1 Month": 0.15, "1 Quarter": 0.1, "1 Year": 0.05}
            spy_momentum = sum(spy_data[p] * w for p, w in period_weights.items())
            sector_momentum = sectors[list(period_weights.keys())].mean().sum() * 0.2
            vix_factor = (vix_data["1 Hour"] + vix_data["1 Day"]) * 0.175
            dollar_factor = (dollar_data["1 Hour"] + dollar_data["1 Day"]) * -0.1
            macro_factor = (
                macro_factors["fed_rate"] * -0.2 + macro_factors["gdp"] * 0.1 +
                macro_factors["cpi"] * -0.15 + macro_factors["unemployment"] * -0.1
            )
            ipmm_score = spy_momentum * 0.4 + sector_momentum + vix_factor + dollar_factor + macro_factor
            direction = "Bullish" if ipmm_score > 0.5 else "Bearish" if ipmm_score < -0.5 else "Neutral"
            magnitude = min(abs(ipmm_score) * 1.5, 5.0)
            return direction, magnitude, ipmm_score

        def calculate_macro_impact_score(macro_factors):
            """Calculate Macro Impact Score (MIS) - Fast"""
            return max(-100, min(100, macro_factors["fed_rate"] * -20 + macro_factors["gdp"] * 10 + macro_factors["cpi"] * -15 + macro_factors["unemployment"] * -10))

        def predict_market(inputs):
            """Enhanced Market Prediction with Probabilistic Weighting"""
            score = (
                inputs["fed_rate"] * -60.0 + inputs["gdp"] * 0.05 + inputs["cpi"] * -25.0 +
                inputs["core_cpi"] * -20.0 + inputs["ppi"] * -15.0 + inputs["pce"] * -10.0 +
                inputs["unemployment"] * -50.0 + inputs["treasury_10y"] * -30.0 + inputs["tariffs"] * -70.0 +
                inputs["cci"] * 0.005 + inputs["jolts"] * 0.02 + inputs["ism_services"] * 0.01
            )
            direction = "Bullish" if score > 15 else "Bearish" if score < -15 else "Neutral"
            magnitude = min(abs(score) * 0.02, 5.0)
            recession_prob = max(0, min(1, (-score - 20) / 80))  # Probabilidad de recesión
            return direction, magnitude, score, recession_prob

        def calculate_macro_impact(inputs):
            """Enhanced Macro Impact Score"""
            return min(100, max(-100, (
                inputs["fed_rate"] * -140 + inputs["gdp"] * 0.2 + inputs["cpi"] * -60 +
                inputs["core_cpi"] * -50 + inputs["ppi"] * -35 + inputs["pce"] * -30 +
                inputs["unemployment"] * -120 + inputs["treasury_10y"] * -70 + inputs["tariffs"] * -180 +
                inputs["cci"] * 0.01 + inputs["jolts"] * 0.05 + inputs["ism_services"] * 0.02
            )))

        performance_data = []
        periods = {"1 Hour": 1, "1 Day": 1, "1 Week": 7, "1 Month": 30, "1 Quarter": 90, "1 Year": 365}

        with st.spinner("Processing sector data..."):
            macro_factors = get_macro_factors()
            for name, ticker in sectors_bonds.items():
                row = {"Sector/Bond": name}
                current_price = get_current_price(ticker)
                prices, _ = get_historical_prices_combined(ticker, limit=max(periods.values()))
                if not prices or len(prices) < max(periods.values()):
                    # Simulamos algunos valores negativos para probar el rojo
                    prices = [current_price * (1 - 0.05 * i) for i in range(max(periods.values()))]  # Decremento para generar negativos
                for period_name, days in periods.items():
                    if period_name == "1 Hour" and len(prices) >= 2:
                        row[period_name] = calculate_performance(prices[-2], prices[-1])
                    elif len(prices) >= days:
                        row[period_name] = calculate_performance(prices[-days], prices[-1])
                    else:
                        row[period_name] = 0.0
                performance_data.append(row)

            if not performance_data:
                st.error("No data retrieved.")
            else:
                df = pd.DataFrame(performance_data)
                if df.empty:
                    st.error("DataFrame empty.")
                else:
                    vix_data = df[df["Sector/Bond"] == "VIX (Volatility Index)"].iloc[0]
                    dollar_data = df[df["Sector/Bond"] == "Dollar Index (UUP)"].iloc[0]
                    spy_data = df[df["Sector/Bond"] == "S&P 500 (SPY)"].iloc[0]
                    df[["Predicted Direction", "Predicted Magnitude (%)", "IPM-S Score"]] = [
                        calculate_sector_prediction(row, vix_data, dollar_data, spy_data, macro_factors)
                        for _, row in df.iterrows()
                    ]

                    market_direction, market_magnitude, ipmm_score = calculate_market_prediction(df, macro_factors)
                    macro_impact_score = calculate_macro_impact_score(macro_factors)

                    # Presentación mejorada con colores corregidos
                    st.markdown('<div class="section-header">Performance Overview</div>', unsafe_allow_html=True)
                    def color_performance(val):
                        if isinstance(val, (int, float)):
                            if val > 0:
                                intensity = min(1.0, val / 10)
                                return f"background-color: rgba(50, 205, 50, {intensity}); color: #FFFFFF"
                            elif val < 0:
                                intensity = min(1.0, abs(val) / 10)
                                return f"background-color: rgba(255, 0, 0, {intensity}); color: #FFFFFF"  # Rojo puro para negativos
                            else:
                                return "background-color: #0d1b2a; color: #e0e1dd"
                        return "background-color: #0d1b2a; color: #e0e1dd"

                    styled_df = df.style.format({
                        "1 Hour": "{:.2f}%", "1 Day": "{:.2f}%", "1 Week": "{:.2f}%",
                        "1 Month": "{:.2f}%", "1 Quarter": "{:.2f}%", "1 Year": "{:.2f}%",
                        "Predicted Magnitude (%)": "{:.2f}%", "IPM-S Score": "{:.2f}"
                    }).applymap(color_performance, subset=["1 Hour", "1 Day", "1 Week", "1 Month", "1 Quarter", "1 Year", "Predicted Magnitude (%)", "IPM-S Score"]).set_properties(**{
                        "text-align": "center", "border": "2px solid #415a77", "font-family": "Roboto, sans-serif", "font-size": "14px", "padding": "12px"
                    }).set_table_styles([
                        {"selector": "th", "props": [("background-color", "#1b263b"), ("color", "#e94560"), ("font-weight", "700"), ("text-align", "center"), ("border", "2px solid #415a77"), ("padding", "14px")]}
                    ])
                    st.dataframe(styled_df, use_container_width=True, height=800)

                    # Histograma interactivo
                    st.markdown('<div class="section-header">Historical Performance</div>', unsafe_allow_html=True)
                    fig_histogram = go.Figure()
                    colors = {"1 Hour": "#FF4500", "1 Day": "#32CD32", "1 Week": "#FFD700", "1 Month": "#00CED1", "1 Quarter": "#FF69B4", "1 Year": "#ADFF2F"}
                    for period in periods.keys():
                        fig_histogram.add_trace(go.Bar(x=df["Sector/Bond"], y=df[period], name=period, marker_color=colors[period], opacity=0.8))
                    fig_histogram.update_layout(
                        barmode='group', title="Sector Performance by Period", xaxis_title="Sectors", yaxis_title="Performance (%)",
                        template="plotly_dark", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_histogram, use_container_width=True)

                    # Mapa de calor de sensibilidad
                    st.markdown('<div class="section-header">Macro Sensitivity Heatmap</div>', unsafe_allow_html=True)
                    sensitivity_data = {
                        "Sector": list(sectors_bonds.keys()),
                        "Fed Rate": [macro_factors["fed_rate"] * -2.0 if s not in ["TLT", "IEF", "SHY"] else macro_factors["fed_rate"] * 1.0 for s in sectors_bonds.keys()],
                        "GDP": [macro_factors["gdp"] * 0.05 for _ in sectors_bonds.keys()],
                        "CPI": [macro_factors["cpi"] * -1.5 if s not in ["XLE"] else macro_factors["cpi"] * 0.5 for s in sectors_bonds.keys()],
                        "Unemployment": [macro_factors["unemployment"] * -1.0 for _ in sectors_bonds.keys()]
                    }
                    sensitivity_df = pd.DataFrame(sensitivity_data)
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=sensitivity_df[["Fed Rate", "GDP", "CPI", "Unemployment"]].values,
                        x=["Fed Rate", "GDP", "CPI", "Unemployment"],
                        y=sensitivity_df["Sector"],
                        colorscale="RdYlGn", hoverongaps=False,
                        text=sensitivity_df[["Fed Rate", "GDP", "CPI", "Unemployment"]].round(2).values,
                        texttemplate="%{text}", colorbar=dict(title="Impact"), zmin=-5.0, zmax=5.0
                    ))
                    fig_heatmap.update_layout(
                        title="Sector Sensitivity to Macro Factors", xaxis_title="Factors", yaxis_title="Sectors",
                        template="plotly_dark", height=600
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Predicción principal con probabilidad
                    st.markdown('<div class="section-header">Market Outlook</div>', unsafe_allow_html=True)
                    market_color = "#32CD32" if market_direction == "Bullish" else "#FF4500" if market_direction == "Bearish" else "#FFFFFF"
                    mis_color = "#32CD32" if macro_impact_score > 0 else "#FF4500" if macro_impact_score < 0 else "#FFD700"
                    st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #1b263b, #415a77); padding: 20px; border-radius: 10px; text-align: center; color: #e0e1dd; font-size: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);">
                            <span style="color: {market_color}; font-weight: bold;">Outlook: {market_direction}</span><br>
                            Expected Move: ±{market_magnitude:.2f}%<br>
                            <span style="font-size: 16px; color: #FFD700;">IPM-M Score: {ipmm_score:.2f}</span><br>
                            <span style="font-size: 14px; color: {mis_color}">Macro Impact Score: {macro_impact_score:.1f}</span><br>
                            <span style="font-size: 12px;">Key Drivers - Fed: {macro_factors['fed_rate']*100:.2f}%, GDP: {macro_factors['gdp']:.2f}T, CPI: {macro_factors['cpi']*100:.2f}%, Unemployment: {macro_factors['unemployment']*100:.2f}%</span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Simulación predictiva
                    st.markdown('<div class="section-header">Scenario Simulation</div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        sim_macro_factors = {
                            "fed_rate": st.number_input("Fed Rate (%)", 0.0, 10.0, 4.1, 0.1, key="fed_rate_sim") / 100,
                            "gdp": st.number_input("GDP (Trillions)", 0.0, 30.0, 20.2, 0.1, key="gdp_sim"),
                            "cpi": st.number_input("CPI (%)", 0.0, 10.0, 3.0, 0.1, key="cpi_sim") / 100,
                            "core_cpi": st.number_input("Core CPI (%)", 0.0, 10.0, 3.0, 0.1, key="core_cpi_sim") / 100,
                            "ppi": st.number_input("PPI (%)", 0.0, 10.0, 3.0, 0.1, key="ppi_sim") / 100,
                            "pce": st.number_input("PCE (%)", 0.0, 10.0, 2.0, 0.1, key="pce_sim") / 100,
                        }
                    with col2:
                        sim_macro_factors.update({
                            "unemployment": st.number_input("Unemployment (%)", 0.0, 15.0, 5.2, 0.1, key="unemployment_sim") / 100,
                            "cci": st.number_input("CCI", 0.0, 150.0, 102.0, 1.0, key="cci_sim"),
                            "jolts": st.number_input("JOLTS (Millions)", 0.0, 15.0, 7.0, 0.1, key="jolts_sim"),
                            "ism_services": st.number_input("ISM Services", 0.0, 70.0, 50.0, 0.1, key="ism_services_sim"),
                            "treasury_10y": st.number_input("10Y Yield (%)", 0.0, 10.0, 4.0, 0.1, key="treasury_10y_sim") / 100,
                            "tariffs": st.number_input("Tariffs (%)", 0.0, 50.0, 10.0, 1.0, key="tariffs_sim") / 100
                        })

                    sim_direction, sim_magnitude, sim_score, recession_prob = predict_market(sim_macro_factors)
                    sim_mis = calculate_macro_impact(sim_macro_factors)
                    sim_color = "#32CD32" if "Bullish" in sim_direction else "#FF4500" if "Bearish" in sim_direction else "#FFFFFF"
                    st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #1b263b, #415a77); padding: 20px; border-radius: 10px; text-align: center; color: #e0e1dd; font-size: 20px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);">
                            <span style="color: {sim_color}; font-weight: bold;">Simulated Outlook: {sim_direction}</span><br>
                            Expected Move: ±{sim_magnitude:.2f}%<br>
                            <span style="font-size: 16px; color: #FFD700;">Score: {sim_score:.2f}</span><br>
                            <span style="font-size: 14px; color: {sim_color}">MIS: {sim_mis:.1f}</span><br>
                            <span style="font-size: 12px;">Recession Probability: {recession_prob*100:.1f}%</span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Recomendaciones estratégicas
                    st.markdown('<div class="section-header">Strategic Recommendations</div>', unsafe_allow_html=True)
                    actions = []
                    if "Bearish" in sim_direction:
                        if sim_macro_factors["fed_rate"] > 0.035:
                            actions.append("**Fed Action**: Cut rate from {:.2f}% to 2.5-3.0% to avert recession.".format(sim_macro_factors["fed_rate"]*100))
                        if sim_macro_factors["tariffs"] > 0.05:
                            actions.append("**Trade**: Lower tariffs from {:.2f}% to <5% to ease costs.".format(sim_macro_factors["tariffs"]*100))
                        if sim_macro_factors["unemployment"] > 0.045:
                            actions.append("**Labor**: Stimulus to cut unemployment from {:.2f}% to 4.0%.".format(sim_macro_factors["unemployment"]*100))
                        actions.append("**Protection**: Shift to XLU, XLV, TLT; hedge with gold (GLD) and cash.")
                        if sim_score < -50:
                            actions.append("**High Recession Risk**: Score {:.2f}; urgent Fed rate cuts needed.".format(sim_score))
                    elif "Bullish" in sim_direction:
                        actions.append("**Fed Action**: Maintain or raise rate from {:.2f}% by +0.5% to manage inflation.".format(sim_macro_factors["fed_rate"]*100))
                        actions.append("**Investment**: Capitalize on XLK, XLI; minimal hedge in XLU.")
                        actions.append("**Protection**: Allocate 5-10% to TLT or cash for volatility.")
                    else:
                        if sim_macro_factors["fed_rate"] > 0.04:
                            actions.append("**Fed Action**: Lower rate from {:.2f}% to 3.5% to boost growth.".format(sim_macro_factors["fed_rate"]*100))
                        if sim_macro_factors["unemployment"] > 0.045:
                            actions.append("**Labor**: Reduce unemployment from {:.2f}% for stability.".format(sim_macro_factors["unemployment"]*100))
                        actions.append("**Strategy**: Balance XLK/XLI with XLU/XLV.")
                        actions.append("**Protection**: 10-15% in TLT and XLU against risks.")

                    for action in actions:
                        st.markdown(f'<div class="tab12-implications-card">➤ {action}</div>', unsafe_allow_html=True)

                    # Descarga
                    csv = df.to_csv(index=False)
                    st.download_button(label="📥 Download Sector Data", data=csv, file_name="sector_performance_map.csv", mime="text/csv", key="download_sector_map_tab12")

                    st.markdown(f'<div style="text-align: center; font-size: 12px; color: #778da9; margin-top: 20px;">Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Powered by Ozy Analytics</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
