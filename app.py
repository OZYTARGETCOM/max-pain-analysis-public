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

# Configurar cliente de Kraken con las claves proporcionadas
API_KEY = "kyFpw+5fbrFIMDuWJmtkbbbr/CgH/MS63wv7dRz3rndamK/XnjNOVkgP"
PRIVATE_KEY = "7xbaBIp902rSBVdIvtfrUNbRHEHMkfMHPEf4rssz+ZwSwjUZFegjdyyYZzcE5DbBrUbtFdGRRGRjTuTnEblZWA=="
kraken = krakenex.API(key=API_KEY, secret=PRIVATE_KEY)

# Configuración de logging
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

def authenticate_password(input_password):
    local_ip = get_local_ip()
    if not local_ip:
        st.error("Could not obtain local IP.")
        logger.error("Failed to obtain local IP during authentication.")
        return False
    passwords = load_passwords()
    for hashed_pwd, data in passwords.items():
        if bcrypt.checkpw(input_password.encode('utf-8'), hashed_pwd.encode('utf-8')):
            if data["usage_count"] < 2:  # Permitir hasta 2 usos
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
    st.error("❌ Incorrect password. If you don’t have access, text 'OzyTarget Access' to 678-978-9414 to purchase your subscription.")
    logger.warning(f"Authentication failed: Invalid password {input_password}")
    return False

def get_local_ip():
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        logger.error("Error obtaining local IP.")
        return None

# Pantalla de autenticación con logo
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
        padding: 20px;
        background: linear-gradient(135deg, #1E1E1E, #2A2A2A);
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    .loading-text {
        font-size: 24px;
        font-weight: bold;
        color: #32CD32;
        text-shadow: 0 0 10px #32CD32;
    }
    .sub-text {
        font-size: 16px;
        color: #FFD700;
        margin-top: 10px;
    }
    .spinner {
        font-size: 30px;
        animation: spin 1.5s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    password = st.text_input("Enter your password", type="password")
    if st.button("LogIn"):
        if not password:
            st.error("❌ Please enter a password.")
        elif authenticate_password(password):
            st.session_state["authenticated"] = True
            # Temporizador regresivo de 7 segundos
            with st.empty():
                for seconds in range(7, 0, -1):
                    st.markdown(f"""
                    <div class="loading-container">
                        <div class="loading-text">✅ ACCESS GRANTED</div>
                        <div class="sub-text">OzyTarget Scanner initializing in {seconds}...</div>
                        <div class="spinner">🔄</div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)  # Espera 1 segundo por cada conteo
                # Mensaje final antes de recargar
                st.markdown("""
                <div class="loading-container">
                    <div class="loading-text">✅ ACCESS GRANTED</div>
                    <div class="sub-text">Deploying OzyTarget Systems Now...</div>
                    <div class="spinner">🔄</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)  # Breve pausa final para efecto
            st.rerun()
    st.stop()
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
            logger.error(f"Error fetching CoinGecko data for {ticker}: {response.status_code}")
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
        logger.error(f"Error fetching CoinGecko data for {ticker}: {str(e)}")
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





# --- Main App --
def main():
    # Logo y título principal después de autenticación
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
          ℙℝ𝕆 𝔼𝕊ℂ𝔸ℕℕ𝔼ℝ|®
        """, unsafe_allow_html=True)
    with col2:
        logo_path = "assets/favicon.png"  # Intenta en assets primero
        if not os.path.exists(logo_path):
            logo_path = "favicon.png"  # Luego en la raíz
        if os.path.exists(logo_path):
            st.image(logo_path, width=45)
        else:
            st.warning("favicon.png' was not found. Please place the file in 'C:/Users/urbin/TradingApp/' or 'assets/'.")
    
    # Estilos personalizado
    st.markdown("""
        <style>
        .stApp {background-color: #1E1E1E;}
        .stTextInput, .stSelectbox {background-color: #2D2D2D; color: #FFFFFF;}
        .stSpinner > div > div {border-color: #32CD32 !important;}
        </style>
    """, unsafe_allow_html=True)

    # Resto de los tabs (agregamos Tab 9)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Gummy Data Bubbles® |", "Market Scanner |", "News |", "Institutional Holders |", 
    "Options Order Flow |", "Analyst Rating Flow |", "Elliott Pulse® |", "Crypto Insights |", 
    "Earnings Calendar |", "Psychological Edge |"
])

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
            max_pain_strike, max_pain_df = calculate_max_pain(df)  # Ahora debería funcionar
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
                    
                    # Descarga CSV
                    csv = pd.DataFrame(results).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Market Scan Data",
                        data=csv,
                        file_name=f"market_scan_{scan_type.replace(' ', '_').lower()}.csv",
                        mime="text/csv",
                        key="download_tab2"
                    )
                else:
                    st.warning("No stocks match the criteria.")
                    st.markdown("---")
                    st.markdown("*Developed by Ozy | © 2025*")

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
                # Descarga CSV
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
                        # Descarga CSV
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

    with tab6:
        st.subheader("Rating Flow")
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
                    # Descarga CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Rating Flow Data",
                        data=csv,
                        file_name=f"{ticker}_rating_flow_{expiration_date}.csv",
                        mime="text/csv",
                        key="download_tab6"
                    )
                else:
                    st.error(f"No alerts generated with Open Interest ≥ {selected_volume}, Gamma ≥ {selected_gamma}. Check logs.")
                    st.markdown("---")
                    st.markdown("*Developed by Ozy | © 2025*")  

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
                                    hovertemplate="Strike: $%{customdata[0]:.2f}<br>Gamma: %{customdata[1]:.2f}<br>OI: %{customdata[2]:,d}<br>Max Pain / Strike: %{customdata[3]:.2f}<br>MM Pressure: %{customdata[4]:.0f}"))
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
            # Descarga CSV
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

    with tab8:
        st.subheader("Crypto Insights")
        
        ticker = st.text_input("Enter Crypto Ticker (e.g., BTC, ETH, XRP):", value="BTC", key="crypto_ticker_tab8").upper()
        selected_pair = f"{ticker}/USD"
        
        refresh_button = st.button("Refresh Orders", key="refresh_tab8")
        
        placeholder = st.empty()
        
        if refresh_button or "tab8_initialized" not in st.session_state:
            with st.spinner(f"Fetching data for {selected_pair}..."):
                try:
                    market_data = fetch_coingecko_data(ticker)
                    if not market_data:
                        st.error(f"Failed to fetch market data for {ticker} from Ozyforward files.")
                    else:
                        bids, asks, current_price = fetch_order_book(ticker, depth=500)
                        if bids.empty or asks.empty:
                            st.error(f"Failed to fetch order book for {selected_pair}. Verify the ticker.")
                        else:
                            with placeholder.container():
                                st.markdown(f"### Bitcoin USD ({ticker}USD)")
                                st.write(f"**Price**: ${market_data['price']:,.2f}")
                                st.write(f"**Change (24h)**: {market_data['change_value']:,.2f} ({market_data['change_percent']:.2f}%)")
                                st.write(f"**Volume (24h)**: {market_data['volume']:,.0f}")
                                st.write(f"**Market Cap**: ${market_data['market_cap']:,.0f}")
                                
                                fig, order_metrics = plot_order_book_bubbles_with_max_pain(bids, asks, current_price, ticker, market_data['volatility'])
                                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_tab8_{ticker}_{int(time.time())}")
                                
                                st.subheader("Key Metrics")
                                pressure_color = "#32CD32" if order_metrics['net_pressure'] > 0 else "#FF4500" if order_metrics['net_pressure'] < 0 else "#FFFFFF"
                                st.write(f"**Net Pressure**: <span style='color:{pressure_color}'>{order_metrics['net_pressure']:,.0f}</span> ({order_metrics['trend']})", unsafe_allow_html=True)
                                st.write(f"**Volatility (Annualized)**: {order_metrics['volatility']:.2f}%")
                                st.write(f"**Projected Target**: ${order_metrics['target_price']:,.2f}")
                                st.write(f"**Support**: ${order_metrics['support']:.2f} | **Resistance**: ${order_metrics['resistance']:.2f}")
                                st.write("**Whale Accumulation Zones**: " + ", ".join([f"${zone:.2f}" for zone in order_metrics['whale_zones']]))
                                edge_color = "#32CD32" if order_metrics['edge_score'] > 50 else "#FF4500" if order_metrics['edge_score'] < 30 else "#FFD700"
                                st.write(f"**Trader's Edge Score**: <span style='color:{edge_color}'>{order_metrics['edge_score']:.1f}</span> (0-100)", unsafe_allow_html=True)
                            
                            st.session_state["tab8_initialized"] = True
                except Exception as e:
                    st.error(f"Error processing data for {selected_pair}: {str(e)}")
                    logger.error(f"Error in Tab 8: {str(e)}")

                    st.markdown("---")
                    st.markdown("*Developed by Ozy | © 2025*")

    with tab9:
        #st.subheader("Earnings Calendar")

        # Rango automático: próxima semana completa (lunes a domingo)
        today = datetime.now().date()
        days_to_next_monday = (7 - today.weekday()) % 7  # Días hasta el próximo lunes
        if days_to_next_monday == 0:  # Si hoy es lunes, saltar a la próxima semana
            days_to_next_monday = 7
        start_date = today + timedelta(days=days_to_next_monday)  # Próximo lunes
        end_date = start_date + timedelta(days=6)  # Próximo domingo

        # Selector de ordenamiento
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
                    logger.warning(f"Invalid date format in earnings: {event_date}")
                    continue
                eps_est = item.get("epsEstimated")
                revenue_est = item.get("revenueEstimated", 0)
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
            
            # Ordenar según criterio
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
            else:  # Date
                earnings.sort(key=lambda x: x["Date"])
            return earnings

        with st.spinner(f"Fetching earnings from {start_date} to {end_date}..."):
            earnings_events = get_earnings_calendar(start_date, end_date)
            logger.info(f"Finished fetching: {len(earnings_events)} events retrieved")

        if earnings_events:
            # Agrupar por día de la semana (sin paginación)
            grouped_events = group_by_day_of_week(earnings_events, start_date, end_date)

            # Diseño de tarjetas con calendario
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
                earnings_html += "<p style='color: #FFFFFF; text-align: center;'>No events to display for next week.</p>"
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

            # Mostrar tarjetas con desplazamiento
            components.html(earnings_html, height=600, scrolling=True)

            # Descarga CSV
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

    with tab10:
        trading_points = [
            ("Sigue los Order Blocks del MM / Spot order blocks of the MM", 
             "Identifica zonas de order blocks (acumulación o distribución del MM) en niveles de liquidez clave. Opera con su flujo, no contra él. Camina antes de la sesión para alinear tu mente con su juego. / Spot order blocks (MM accumulation or distribution) at key liquidity zones. Trade with their flow, not against it. Walk before the session to sync your mind with their game.", 
             "#2A2A3D"),  # Gris azulado oscuro
            ("Domina el Riesgo Institucional / Master Institutional Risk", 
             "Limita el riesgo al 1-2% por posición, ajustado por VaR (Value at Risk) si operas portafolios grandes. No entres si tu estado emocional compromete el análisis de datos duros. / Cap risk at 1-2% per position, adjusted by VaR for large portfolios. Don’t trade if your emotional state clouds hard data analysis.", 
             "#1A2E2A"),  # Verde oscuro
            ("Analiza la Exposición al Gamma / Analyze Gamma Exposure", 
             "Monitorea el gamma de opciones para prever cambios bruscos en el delta. Anticipa trampas del MM cerca de vencimientos (gamma squeezes) y reflexiona: ¿dónde están cazando stops? / Track options gamma to predict sharp delta shifts. Anticipate MM traps near expirations (gamma squeezes) and ask: Where are they hunting stops?", 
             "#3D2A2A"),  # Marrón oscuro
            ("Stop-Loss Basado en Liquidez / Liquidity-Based Stop-Loss", 
             "Coloca stops en niveles de liquidez del MM (debajo de soportes o encima de resistencias), no en zonas aleatorias. Corta rápido: en trading institucional, preservar capital es prioridad. Opera de pie para máxima alerta. / Set stops at MM liquidity levels (below support or above resistance), not random zones. Cut fast: in institutional trading, capital preservation is king. Trade standing for peak alertness.", 
             "#2A2A3D"),  # Gris azulado oscuro
            ("Confirma con Flujo Institucional / Confirm with Institutional Flow", 
             "Busca order flow en order blocks con picos de volumen y gamma. Usa herramientas como footprint charts o monitores de bloques si tienes acceso. Pregúntate: ¿qué están acumulando o descargando? / Look for order flow in order blocks with volume and gamma spikes. Use footprint charts or block monitors if available. Ask: What are they accumulating or unloading?", 
             "#1E2A3D"),  # Azul oscuro
            ("Controla la Psicología bajo Presión / Control Psychology Under Pressure", 
             "Evita reaccionar a volatilidad intradía o ruido de mercado. La disciplina del MM es tu modelo: no te sobreapalances ni cedas a la euforia. Muévete cada hora para no quemarte. / Don’t react to intraday volatility or market noise. MM discipline is your model: no over-leverage or euphoria. Move hourly to avoid burnout.", 
             "#2E2A1E"),  # Marrón grisáceo oscuro
            ("Simula con Datos Reales / Simulate with Real Data", 
             "Usa simulación con datos históricos de order flow y gamma para replicar movimientos del MM. Analiza cómo tu toma de decisiones resiste bajo presión institucional. / Simulate with historical order flow and gamma data to mirror MM moves. Test how your decision-making holds under institutional pressure.", 
             "#1A2E2A"),  # Verde oscuro
            ("Sigue el Volumen Institucional / Track Institutional Volume", 
             "Alto volumen en niveles clave con cambios en gamma confirma compromiso del MM. Opera con ellos cuando el smart money entra; usa volume profile para precisión. Camina para mantener claridad. / High volume at key levels with gamma shifts confirms MM commitment. Trade with them when smart money steps in; use volume profile for precision. Walk to stay clear-headed.", 
             "#2A2A3D"),  # Gris azulado oscuro
            ("Registra Cada Movimiento / Log Every Move", 
             "Documenta trades con niveles de order blocks, gamma, volumen y resultados. Incluye tu estado mental para ajustar sesgos. Los institucionales viven de datos, no de intuición. / Log trades with order blocks, gamma, volume, and outcomes. Add your mental state to tweak biases. Institutionals live on data, not gut feel.", 
             "#1E2A3D"),  # Azul oscuro
            ("Sal en Zonas de Liquidez / Exit at Liquidity Zones", 
             "Toma ganancias en resistencias del MM o cuando el gamma indique un reversal. Usa liquidity grabs a tu favor: el MM a menudo empuja precios para atrapar a los débiles. / Take profits at MM resistance or when gamma signals a reversal. Leverage liquidity grabs: MM often pushes prices to trap the weak.", 
             "#3D2A2A"),  # Marrón oscuro
            ("Filtra el Ruido del Retail / Filter Retail Noise", 
             "Ignora hype de redes sociales o noticias sin respaldo en order flow. Los institucionales confían en el DOM (Depth of Market) y el precio, no en titulares. / Ignore retail hype on social media or news without order flow backing. Institutionals trust DOM (Depth of Market) and price, not headlines.", 
             "#2E2A1E"),  # Marrón grisáceo oscuro
            ("Ajusta por Volatilidad / Adjust for Volatility", 
             "En días de alta volatilidad (vencimientos, eventos macro), reduce tamaño de posición. En días tranquilos, busca order blocks profundos para entradas sólidas. Adapta como el MM. / On high-volatility days (expirations, macro events), shrink position size. On quiet days, target deep order blocks for strong entries. Adapt like the MM.", 
             "#1A2E2A"),  # Verde oscuro
            ("Explota las Ineficiencias / Exploit Inefficiencies", 
             "Busca desbalances entre order flow y gamma (ej. short squeezes o stop runs). Los institucionales ganan donde el retail pierde: opera con ventaja, no con esperanza. / Hunt imbalances between order flow and gamma (e.g., short squeezes or stop runs). Institutionals win where retail loses: trade with edge, not hope.", 
             "#2A2A3D"),  # Gris azulado oscuro
            ("Piensa como el MM / Think Like the MM", 
             "Pregúntate: ¿dónde colocan liquidez para atraer volumen? Usa su lógica (atrapar stops, forzar reversals) para anticipar y alinearte. Reflexiona entre sesiones. / Ask: Where do they place liquidity to draw volume? Use their logic (trap stops, force reversals) to anticipate and align. Reflect between sessions.", 
             "#1E2A3D"),  # Azul oscuro
            ("Monitorea el Gummy Data Bubbles® y la Volatilidad Implícita / Monitor Gummy Data Bubbles® and Implied Volatility", 
             "Analiza el skew de opciones (asimetría en volatilidad implícita) para detectar sesgos del MM hacia alzas o bajas. Un skew pronunciado puede señalar liquidity grabs o stop hunts. Usa esta data para afinar entradas. / Track options skew (implied volatility asymmetry) to spot MM bias toward upside or downside. Sharp skew can signal liquidity grabs or stop hunts. Use it to fine-tune entries.", 
             "#3D2A2A"),  # Marrón oscuro
            ("Aprovecha el Dark Pool Flow / Leverage Dark Pool Flow", 
             "Si tienes acceso Monitores de Ozytarget (volumen institucional oculto), busca confirmación de order blocks. El MM usa estos flujos para mover mercados sin alertar al retail. Alinea tus trades con este smart money. / If you have dark pool data (hidden institutional volume), confirm order blocks. MM uses these flows to move markets without tipping off retail. Align trades with this smart money.", 
             "#2E2A1E"),  # Marrón grisáceo oscuro
            ("Juega el Juego del Spoofing Legal / Play the Legal Spoofing Game", 
             "Detecta patrones de spoofing  en el DOM o tape. No luches contra ellos; úsalos para entrar cuando el precio revierta tras el flush de liquidez. Requiere velocidad y precisión. / Spot spoofing patterns (MM fake orders to mislead) in the DOM or tape. Don’t fight them; use them to enter when price reverses after the liquidity flush. Demands speed and precision.", 
             "#1A2E2A")  # Verde oscuro
        ]

        for i, (title, content, color) in enumerate(trading_points, 1):
            with st.expander(f"{i}. {title}"):
                st.markdown(f"""
                <div style='background-color: {color}; padding: 10px; border-radius: 5px; color: #FFFFFF;'>
                    {content}
                </div>
                """, unsafe_allow_html=True)

        # Sección de Inversión a Largo Plazo
        st.subheader("Inversión a largo plazo para institucionales / Long-Term Investing for Institutionals")
        
        investing_points = [
            ("Horizontes por Edad y Objetivo / Horizons by Age and Goal", 
             "Invierte según el ciclo: 5-10 años para capital activo, 15-20 para crecimiento sostenido, 30+ para legado o fondos soberanos. Elige activos con fundamentales sólidos, no especulación. / Invest by cycle: 5-10 years for active capital, 15-20 for sustained growth, 30+ for legacy or sovereign funds. Pick assets with solid fundamentals, not speculation.", 
             "#2A2A3D"),  # Gris azulado oscuro
            ("Diversifica con Precisión / Diversify with Precision", 
             "Reparte entre clases de activos (acciones, bonos, materias primas) según correlaciones y riesgo ajustado (Sharpe ratio). Compra más en caídas si el order flow institucional lo respalda. / Spread across asset classes (stocks, bonds, commodities) by correlations and risk-adjusted return (Sharpe ratio). Buy more on dips if institutional order flow supports it.", 
             "#1A2E2A"),  # Verde oscuro
            ("Foco en Valor Fundamental / Focus on Fundamental Value", 
             "Selecciona empresas con flujo de caja robusto, baja deuda y ventaja estructural. Ignora ruido de corto plazo: los institucionales miran décadas, no días. / Choose firms with strong cash flow, low debt, and structural edge. Ignore short-term noise: institutionals eye decades, not days.", 
             "#3D2A2A"),  # Marrón oscuro
            ("Reinversión Estratégica / Strategic Reinvestment", 
             "Usa ganancias para aumentar exposición en activos con smart money detrás. Aprovecha caídas para acumular: el MM exagera el miedo para comprar barato. / Reinvest profits to boost exposure to smart money-backed assets. Capitalize on dips: MM overplays fear to buy low.", 
             "#1E2A3D"),  # Azul oscuro
            ("Paciencia Institucional / Institutional Patience", 
             "Mantén posiciones a través de ciclos económicos: el crecimiento global supera las crisis (dato histórico). Compra más en rojo con análisis de order flow, no pánico. / Hold through economic cycles: global growth outlasts crises (historical fact). Buy more in the red with order flow analysis, not panic.", 
             "#2E2A1E"),  # Marrón grisáceo oscuro
            ("Visión Macro Avanzada / Advanced Macro Vision", 
             "Invierte en sectores con respaldo institucional (tecnología, infraestructura, energía limpia) según datos macro (PIB, tasas, demografía). Ajusta con calma, no por modas. / Invest in institutionally backed sectors (tech, infrastructure, clean energy) based on macro data (GDP, rates, demographics). Adjust calmly, not on trends.", 
             "#2A2A3D"),  # Gris azulado oscuro
            ("Contrario con Datos / Contrarian with Data", 
             "Aumenta posiciones en caídas si el volume profile y fundamentales lo justifican. El MM usa el pánico retail para acumular; tú haz lo mismo con ventaja. / Build positions on dips if volume profile and fundamentals hold. MM uses retail panic to accumulate; you do the same with an edge.", 
             "#1A2E2A"),  # Verde oscuro
            ("Gestión Activa Pasiva / Active-Passive Management", 
             "Combina tenencia pasiva (ETFs, índices) con compras activas en order blocks de largo plazo. Rebalancea según smart money, no emociones. / Blend passive holding (ETFs, indexes) with active buys at long-term order blocks. Rebalance by smart money, not emotion.", 
             "#3D2A2A"),  # Marrón oscuro
            ("Incorpora Derivados para Cobertura / Use Derivatives for Hedging", 
             "Usa futuros o opciones para cubrir portafolios contra caídas sin vender activos clave. Los institucionales protegen ganancias sin sacrificar exposición a largo plazo. Ajusta según gamma y volatility index (VIX). / Use futures or options to hedge portfolios against dips without selling core assets. Institutionals shield gains without losing long-term exposure. Adjust by gamma and volatility index (VIX).", 
             "#1E2A3D"),  # Azul oscuro
            ("Explota Ciclos de Rebalancing / Exploit Rebalancing Cycles", 
             "Aprovecha ventanas de rebalancing institucional (fin de mes, trimestre) cuando el MM ajusta posiciones. Compra en caídas inducidas por estos flujos; el volume profile te dirá dónde entran. / Exploit institutional rebalancing windows (month-end, quarter-end) when MM adjusts positions. Buy dips driven by these flows; volume profile shows where they step in.", 
             "#2E2A1E")  # Marrón grisáceo oscuro
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

if __name__ == "__main__":
    main()
