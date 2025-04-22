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
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from datetime import datetime, timedelta
import multiprocessing
from scipy import stats
import plotly.express as px
import csv
import bcrypt
import sqlite3
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import socket
from scipy.stats import norm
import xml.etree.ElementTree as ET
import streamlit.components.v1 as components
import krakenex
import base64
import threading


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

# Constantes
PASSWORDS_DB = "auth_data/passwords.db"
CACHE_TTL = 300
MAX_RETRIES = 5
INITIAL_DELAY = 1
RISK_FREE_RATE = 0.045  # Definimos RISK_FREE_RATE aquí

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración inicial de página (DEBE SER LA PRIMERA LLAMADA DE STREAMLIT)
st.set_page_config(
    page_title="Pro Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Autenticación con SQLite ---
def initialize_passwords_db():
    os.makedirs("auth_data", exist_ok=True)
    conn = sqlite3.connect(PASSWORDS_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS passwords 
                 (password TEXT PRIMARY KEY, usage_count INTEGER DEFAULT 0, ip1 TEXT DEFAULT '', ip2 TEXT DEFAULT '')''')
    initial_passwords = [
        ("abc234", 0, "", ""), ("def456", 0, "", ""), ("ghi789", 0, "", ""),
        ("jkl010", 0, "", ""), ("mno345", 0, "", ""), ("pqr678", 0, "", ""),
        ("stu901", 0, "", ""), ("vwx234", 0, "", ""), ("yz1234", 0, "", ""),
        ("abcd56", 0, "", ""), ("efgh78", 0, "", ""), ("ijkl90", 0, "", ""),
        ("mnop12", 0, "", ""), ("qrst34", 0, "", ""), ("uvwx56", 0, "", ""),
        ("yzab78", 0, "", ""), ("cdef90", 0, "", ""), ("ghij12", 0, "", ""),
        ("news34", 0, "", ""), ("opqr56", 0, "", ""), ("xyz789", 0, "", ""),
        ("kml456", 0, "", ""), ("nop123", 0, "", ""), ("qwe987", 0, "", ""),
        ("asd654", 0, "", ""), ("zxc321", 0, "", ""), ("bnm098", 0, "", ""),
        ("vfr765", 0, "", ""), ("tgb432", 0, "", ""), ("hju109", 0, "", "")
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
                st.error("❌ This password has already been used by two IPs. To get your own access to Pro Scanner, text 'Pro Scanner Access' to 678-978-9414.")
                logger.warning(f"Authentication attempt for {input_password} from IP {local_ip} rejected; already used from {data['ip1']} and {data['ip2']}")
                return False
    st.error("❌ Incorrect password. If you don’t have access, text 'Pro Scanner Access' to 678-978-9414 to purchase your subscription.")
    logger.warning(f"Authentication failed: Invalid password {input_password}")
    return False

# Inicializar la base de datos
initialize_passwords_db()


# Estado de la sesión
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "intro_shown" not in st.session_state:
    st.session_state["intro_shown"] = False

# Animación introductoria estilo CMD con hackeo
if not st.session_state["intro_shown"]:
    st.markdown("""
    <style>
    /* Fondo negro para la intro */
    .stApp {
        background-color: #000000;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    .intro-container {
        width: 100vw; /* Ancho completo */
        height: 100vh; /* Alto completo */
        background: #000000;
        border: 2px solid #FFFF00; /* Amarillo */
        padding: 40px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 20px; /* Tamaño de fuente aumentado */
        line-height: 1.5;
        white-space: pre-wrap;
        overflow-y: auto;
        box-shadow: 0 0 20px rgba(255, 255, 0, 0.5); /* Sombra amarilla */
        position: fixed;
        top: 0;
        left: 0;
        z-index: 9999;
    }
    .intro-text {
        animation: typing 4s steps(40, end); /* Reducido de 7s a 4s para mayor velocidad */
        display: inline-block;
    }
    .yellow { color: #FFFF00; } /* Amarillo */
    .green { color: #39FF14; } /* Verde neón */
    .red { color: #FF0000; } /* Rojo */
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-container" id="introContainer">
        <div class="intro-text">
<span class="yellow">> INITIALIZING PRO SCANNER v3.0.0...</span>
<span class="green">> LOADING HACKING MODULES...</span>
[OK] Bypassing Market Maker (MM) Firewalls...
[OK] Accessing MM Liquidity Pools...
<span class="red">[ALERT] MM Defense Systems Detected - Countermeasures Deployed</span>
[OK] Extracting EBITDA Data from Corporate Servers...
<span class="yellow">[PROGRESS] 25%... 50%... 75%... 100%</span>
[OK] EBITDA Data Compromised
<span class="green">> INFILTRATING BROKER NETWORKS...</span>
[OK] Nasdq API Breached Bypassing
[OK] NYSE API Breached  Bypassing
<span class="red">[ALERT] Broker Counter-Hack Attempt - Neutralized</span>
[OK] Financial Data Streams Intercepted
<span class="yellow">> SYSTEM CHECK COMPLETE</span>
<span class="green">> STATUS: READY FOR DEPLOYMENT</span>
<span class="red">>CMD WINDOW : authorized Access Detected - Enter Credentials to Proceed...</span>
        </div>
    </div>
    <script>
    setTimeout(function() {
        document.getElementById('introContainer').style.display = 'none';
    }, 4000); /* Reducido de 7000ms a 4000ms */
    </script>
    """, unsafe_allow_html=True)

    time.sleep(4)  # Duración de la animación introductoria (reducida de 7s a 4s)
    st.session_state["intro_shown"] = True
    st.rerun()

# Pantalla de login original
if not st.session_state["authenticated"]:
    st.markdown("""
    <style>
    /* Fondo global negro puro */
    .stApp {
        background-color: #000000;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        min-height: 100vh;
        margin: 0;
        padding: 0;
    }
    /* Eliminar cualquier contenedor superior */
    .st-emotion-cache-1gv3huu {
        display: none;
    }
    .login-container {
        padding: 20px;
        text-align: center;
        margin-top: 25vh;
        position: relative;
        z-index: 10;
    }
    .login-logo {
        font-size: 18px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 15px;
        letter-spacing: 1px;
        position: relative;
        z-index: 10;
    }
    /* Estilo del formulario (el recuadro que rodea el input y botón) */
    div.stForm {
        border: 2px solid #39FF14 !important; /* Borde neón verde */
        border-radius: 5px !important;
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.5) !important; /* Sombra neón */
        background: rgba(0, 0, 0, 0.1) !important; /* Fondo ligeramente transparente */
    }
    .login-input {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 2px solid #39FF14 !important;
        border-radius: 5px;
        padding: 3px;
        width: 50px !important;
        font-size: 6px;
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.5);
        position: relative;
        z-index: 10;
    }
    .login-button {
        background-color: #FFFFFF;
        color: #000000;
        padding: 3px 6px;
        border: 2px solid #39FF14 !important;
        border-radius: 5px;
        font-size: 6px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 50px !important;
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.5);
        position: relative;
        z-index: 10;
    }
    .login-button:hover {
        background-color: #E0E0E0;
    }
    .hacker-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }
    .hacker-text {
        font-size: 24px;
        font-weight: 700;
        color: #FFFF00; /* Amarillo */
        text-shadow: 0 0 15px #FFFF00;
        letter-spacing: 2px;
        position: relative;
        z-index: 10000;
    }
    .hacker-canvas {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 9998;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-logo">ℙℝ𝕆 𝔼𝕊ℂ𝔸ℕℕ𝔼ℝ®</div>', unsafe_allow_html=True)

        with st.form(key="login_form"):
            password = st.text_input("", type="password", key="login_input", placeholder="Password")
            submit_button = st.form_submit_button(label="Log In")

            if submit_button:
                if not password:
                    st.error("❌ Please enter a password.")
                elif authenticate_password(password):
                    st.session_state["authenticated"] = True
                    st.markdown("""
                    <div class="-overlay" id="hackerOverlay">
                        <canvas class="hacker-canvas" id="hackerCanvas"></canvas>
                        <div class="hacker-text">✅ ACCESS GRANTED</div>
                    </div>
                    <script>
                    const [OK] Nasdq API Breached Bypassing = document.getElementById('hackerCanvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = window.innerWidth;
                    canvas.height = [OK] Nasdq API Breached Bypassing;

                    const numbers = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789{}[]()+-*/=<>;,.#$%&@!'; /*  RISK_FREE_RATE */
                    const fontSize = 20;
                    const columns = can.width / fontSize;
                    const drops = [];

                    for (let x = 0; x < columns; x++) {
                        drops[x] = Math.random() *  RISK_FREE_RATE;
                    }

                    function drawDynamic() {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = '#FFFF00'; /* Amarillo */
                        ctx.shadowBlur = 20;
                        ctx.shadowColor = '#FFFF00';
                        ctx.font = fontSize + 'px monospace';

                        for (let i = 0; i < drops.length; i++) {
                            const text = numbers.charAt(Math.floor(Math.random() * numbers.length));
                            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                            drops[i] += 8; /* [OK] Nasdq API Breached Bypassing (de 5 a 8) */
                            if (drops[i] * fontSize > canvas.height && Math.random() > 0.95) {
                                drops[i] = -fontSize;
                            }
                        }
                        ctx.shadowBlur = 0;
                        requestAnimationFrame(drawDynamic);
                    }

                    function drawStatic() {
                        ctx.fillStyle = '#39FF14'; /* Verde */
                        ctx.shadowBlur = 20;
                        ctx.shadowColor = '#39FF14';
                        ctx.font = fontSize + 'px monospace';
                        for (let i = 0; i < columns; i++) {
                            const text = numbers.charAt(Math.floor(Math.random() * numbers.length));
                            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                        }
                    }

                    try {
                        drawDynamic();
                    } catch (e) {
                        drawStatic();
                    }

                    setTimeout(function() {
                        document.getElementById('Overlay').style.display = 'none';
                    }, 1000); /* Reducido de 2000ms a 1000ms */
                    </script>
                    """, unsafe_allow_html=True)
                    time.sleep(1)  # Reducido de 2s a 1s
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
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

@st.cache_data(ttl=300)
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

@st.cache_data(ttl=300)
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












@st.cache_data(ttl=3600)
def get_options_data(ticker: str, expiration_date: str) -> List[Dict]:
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {"symbol": ticker, "expiration": expiration_date, "greeks": "true"}
    data = fetch_api_data(url, params, HEADERS_TRADIER, "Tradier")
    if data and 'options' in data and 'option' in data['options']:
        return data['options']['option']
    return []

@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=3600)
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
def analyze_contracts(ticker, expiration, current_price, options_data=None):
    """
    Analyzes option contracts for a given ticker and expiration date.
    Uses provided options_data if available, otherwise fetches from API.
    Returns a DataFrame with contract details.
    """
    if options_data is not None:
        # Use the provided options_data
        if not options_data:
            logger.warning(f"No options data provided for {ticker} on {expiration} in analyze_contracts.")
            return pd.DataFrame(columns=["strike", "option_type", "open_interest", "volume", "bid"])
        
        df = pd.DataFrame(options_data)
        if df.empty:
            return pd.DataFrame(columns=["strike", "option_type", "open_interest", "volume", "bid"])
        
        # Ensure option_type is lowercase for consistency
        df["option_type"] = df["option_type"].str.lower()
        return df[["strike", "option_type", "open_interest", "volume", "bid"]]
    
    # Fallback to API call if no options_data provided
    url = f"{TRADIER_BASE_URL}/markets/options/chains"
    params = {
        "symbol": ticker,
        "expiration": expiration,
        "greeks": "true"
    }
    try:
        response = requests.get(url, headers=HEADERS_TRADIER, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data is None:
            logger.warning(f"No JSON data returned from API for {ticker} on {expiration}. Response: {response.text}")
            return pd.DataFrame(columns=["strike", "option_type", "open_interest", "volume", "bid"])
        
        options = data.get("options", {}).get("option", [])
        if not options:
            logger.warning(f"No options data found in analyze_contracts for {ticker} on {expiration}. Response: {data}")
            return pd.DataFrame(columns=["strike", "option_type", "open_interest", "volume", "bid"])
        
        df = pd.DataFrame(options)
        if df.empty:
            return pd.DataFrame(columns=["strike", "option_type", "open_interest", "volume", "bid"])
        
        df["option_type"] = df["option_type"].str.lower()
        return df[["strike", "option_type", "open_interest", "volume", "bid"]]
    except Exception as e:
        logger.error(f"Error fetching contracts for {ticker} on {expiration}: {str(e)}")
        return pd.DataFrame(columns=["strike", "option_type", "open_interest", "volume", "bid"])

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
    """Calculates the Max Pain for options."""
    if df.empty or 'strike' not in df.columns:
        logger.error("DataFrame is empty or missing 'strike' column in calculate_max_pain")
        return None, pd.DataFrame(columns=['strike', 'total_loss'])

    strikes = df['strike'].unique()
    if len(strikes) == 0:
        logger.error("No unique strikes found in DataFrame")
        return None, pd.DataFrame(columns=['strike', 'total_loss'])

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
        logger.warning("No Max Pain data generated")
        return None, max_pain_df

    # Find the index of the minimum total loss
    min_loss_idx = max_pain_df['total_loss'].idxmin()
    # Extract the strike value as a float
    max_pain_strike = float(max_pain_df.loc[min_loss_idx, 'strike'])
    
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
    # Extraer datos básicos de options_data
    strikes = [float(option["strike"]) for option in options_data]
    iv = [float(option.get("implied_volatility", 0)) * 100 for option in options_data]
    option_type = [option["option_type"].upper() for option in options_data]
    # Asegurarse de que open_interest sea numérico y no nan
    open_interest = [int(option.get("open_interest", 0) or 0) for option in options_data]
    
    # Calcular totales
    total_calls = sum(oi for oi, ot in zip(open_interest, option_type) if ot == "CALL")
    total_puts = sum(oi for oi, ot in zip(open_interest, option_type) if ot == "PUT")
    total_volume_calls = sum(int(option.get("volume", 0)) for option in options_data if option["option_type"].upper() == "CALL")
    total_volume_puts = sum(int(option.get("volume", 0)) for option in options_data if option["option_type"].upper() == "PUT")
    
    # Calcular IV ajustada
    adjusted_iv = [iv[i] + (open_interest[i] * 0.01) if option_type[i] == "CALL" else -(iv[i] + (open_interest[i] * 0.01)) for i in range(len(iv))]
    
    # Crear DataFrame y limpiar datos
    skew_df = pd.DataFrame({
        "Strike": strikes,
        "Adjusted IV (%)": adjusted_iv,
        "Option Type": option_type,
        "Open Interest": open_interest
    })
    # Reemplazar nan con 0 y asegurar valores no negativos
    skew_df["Open Interest"] = skew_df["Open Interest"].fillna(0).astype(int).clip(lower=0)
    
    # Crear gráfico de dispersión con tamaño limpio
    fig = px.scatter(
        skew_df,
        x="Strike",
        y="Adjusted IV (%)",
        color="Option Type",
        size="Open Interest",
        size_max=30,  # Limitar tamaño máximo para mejor visualización
        custom_data=["Strike", "Option Type", "Open Interest", "Adjusted IV (%)"],
        title=f"IV Analysis<br><span style='font-size:16px;'> CALLS: {total_calls} | PUTS: {total_puts} | VC {total_volume_calls} | VP {total_volume_puts}</span>",
        labels={"Option Type": "Contract Type"},
        color_discrete_map={"CALL": "blue", "PUT": "red"}
    )
    fig.update_traces(
        hovertemplate="<b>Strike:</b> %{customdata[0]:.2f}<br><b>Type:</b> %{customdata[1]}<br><b>Open Interest:</b> %{customdata[2]:,}<br><b>Adjusted IV:</b> %{customdata[3]:.2f}%"
    )
    fig.update_layout(
        xaxis_title="Strike Price",
        yaxis_title="Gummy Bubbles® (%)",
        legend_title="Option Type",
        template="plotly_white",
        title_x=0.5
    )

    # Lógica para current_price y max_pain (sin cambios en esta parte)
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
                fig.add_scatter(
                    x=[current_price],
                    y=[avg_iv_calls],
                    mode="markers",
                    name="Current Price (CALLs)",
                    marker=dict(size=call_size, color="yellow", opacity=0.45, symbol="circle"),
                    hovertemplate=(f"Current Price (CALLs): {current_price:.2f}<br>"
                                   f"Adjusted IV: {avg_iv_calls:.2f}%<br>"
                                   f"Open Interest: {call_open_interest:,}<br>"
                                   f"% to Max Pain: {percent_change_calls:.2f}%<br>"
                                   f"R/R: {rr_calls:.2f}<br>"
                                   f"Est. Loss: ${call_loss:,.2f}<br>"
                                   f"Potential Move: ${potential_move_calls:.2f}<br>"
                                   f"Direction: {direction_calls}")
                )

            if put_open_interest > 0 and closest_put:
                fig.add_scatter(
                    x=[current_price],
                    y=[avg_iv_puts],
                    mode="markers",
                    name="Current Price (PUTs)",
                    marker=dict(size=put_size, color="yellow", opacity=0.45, symbol="circle"),
                    hovertemplate=(f"Current Price (PUTs): {current_price:.2f}<br>"
                                   f"Adjusted IV: {avg_iv_puts:.2f}%<br>"
                                   f"Open Interest: {put_open_interest:,}<br>"
                                   f"% to Max Pain: {percent_change_puts:.2f}%<br>"
                                   f"R/R: {rr_puts:.2f}<br>"
                                   f"Est. Loss: ${put_loss:,.2f}<br>"
                                   f"Potential Move: ${potential_move_puts:.2f}<br>"
                                   f"Direction: {direction_puts}")
                )

        if max_pain is not None:
            fig.add_scatter(
                x=[max_pain],
                y=[0],
                mode="markers",
                name="Max Pain",
                marker=dict(size=15, color="white", symbol="circle"),
                hovertemplate=f"Max Pain: {max_pain:.2f}"
            )

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


# Funciones de análisis de sentimiento (agregadas aquí para evitar NameError)
def calculate_retail_sentiment(news):
    """Calcula el sentimiento de mercado de retail basado en titulares de noticias."""
    if not news:
        return 0.5, "Neutral"  # Valor por defecto si no hay noticias
    
    positive_keywords = ["up", "bullish", "gain", "rise", "surge", "strong", "rally", "positive", "growth"]
    negative_keywords = ["down", "bearish", "loss", "drop", "fall", "crash", "weak", "decline", "negative"]
    
    sentiment_score = 0
    total_articles = len(news)
    
    for article in news:
        title = article["title"].lower()
        positive_count = sum(1 for word in positive_keywords if word in title)
        negative_count = sum(1 for word in negative_keywords if word in title)
        sentiment_score += (positive_count - negative_count)
    
    max_possible_score = max(total_articles, 1)
    normalized_score = (sentiment_score + max_possible_score) / (2 * max_possible_score)
    normalized_score = max(0, min(1, normalized_score))
    
    if normalized_score > 0.7:
        sentiment_text = "Very Bullish"
    elif normalized_score > 0.5:
        sentiment_text = "Bullish"
    elif normalized_score < 0.3:
        sentiment_text = "Very Bearish"
    elif normalized_score < 0.5:
        sentiment_text = "Bearish"
    else:
        sentiment_text = "Neutral"
    
    return normalized_score, sentiment_text

def calculate_volatility_sentiment(news):
    """Calcula el sentimiento de volatilidad basado en titulares de noticias."""
    if not news:
        return 0, "Stable"  # Valor por defecto si no hay noticias
    
    high_volatility_keywords = ["crash", "surge", "volatile", "plunge", "spike", "wild", "turmoil", "shock", "boom"]
    low_volatility_keywords = ["steady", "calm", "stable", "flat", "unchanged", "quiet", "consistent"]
    
    volatility_score = 0
    total_articles = len(news)
    
    for article in news:
        title = article["title"].lower()
        high_vol_count = sum(1 for word in high_volatility_keywords if word in title)
        low_vol_count = sum(1 for word in low_volatility_keywords if word in title)
        volatility_score += (high_vol_count - low_vol_count)
    
    max_possible_score = max(total_articles, 1)
    normalized_score = (volatility_score + max_possible_score) / (2 * max_possible_score) * 100
    normalized_score = max(0, min(100, normalized_score))
    
    if normalized_score > 75:
        volatility_text = "Very High Volatility"
    elif normalized_score > 50:
        volatility_text = "High Volatility"
    elif normalized_score < 25:
        volatility_text = "Low Volatility"
    elif normalized_score < 50:
        volatility_text = "Moderate Volatility"
    else:
        volatility_text = "Stable"
    
    return normalized_score, volatility_text








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
    try:
        response = requests.get(url, headers=HEADERS_TRADIER, params=params, timeout=10)
        if response.status_code != 200:
            st.error(f"Error al obtener los datos de opciones. Código de estado: {response.status_code}")
            logger.error(f"API request failed for {symbol} with expiration {expiration_date}: Status {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
        if data is None or not isinstance(data, dict):
            st.error(f"Datos de opciones inválidos para {symbol}. Respuesta vacía o no JSON.")
            logger.error(f"Invalid JSON response for {symbol}: {response.text}")
            return pd.DataFrame()
        
        if 'options' in data and isinstance(data['options'], dict) and 'option' in data['options']:
            options = data['options']['option']
            if not options:
                st.warning(f"No se encontraron contratos de opciones para {symbol} en {expiration_date}.")
                logger.info(f"No option contracts found for {symbol} on {expiration_date}")
                return pd.DataFrame()
            df = pd.DataFrame(options)
            df['action'] = df.apply(lambda row: "buy" if (row.get("bid", 0) > 0 and row.get("ask", 0) > 0) else "sell", axis=1)
            return df
        
        st.error(f"No se encontraron datos de opciones válidos en la respuesta para {symbol}.")
        logger.error(f"Options data missing or malformed for {symbol}: {data}")
        return pd.DataFrame()
    
    except requests.RequestException as e:
        st.error(f"Error de red al obtener datos de opciones para {symbol}: {str(e)}")
        logger.error(f"Network error fetching options for {symbol}: {str(e)}")
        return pd.DataFrame()
    except ValueError as e:
        st.error(f"Error al procesar la respuesta JSON para {symbol}: {str(e)}")
        logger.error(f"JSON parsing error for {symbol}: {str(e)}")
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
    """Obtiene datos macroeconómicos recientes desde FMP con validaciones robustas."""
    url = f"{FMP_BASE_URL}/economic?name={indicator}"
    params = {"apikey": FMP_API_KEY}
    try:
        data = fetch_api_data(url, params, HEADERS_FMP, f"FMP Macro {indicator}")
        if data and isinstance(data, list) and len(data) > 0 and "value" in data[0]:
            value = float(data[0]["value"])
            # Ajustar según la unidad del indicador
            if indicator in ["CPI", "CORE_CPI", "PPI", "PCE", "FEDFUNDS", "UNEMPLOYMENT"]:
                value /= 100  # Convertir de porcentaje a decimal
            elif indicator == "GDP":
                value /= 1_000_000_000_000  # Convertir de billones a trillones
            logger.info(f"{indicator}: {value}")
            return value
        else:
            raise ValueError(f"No valid data for {indicator}")
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"Error fetching {indicator} data: {str(e)}. Using fallback.")
        fallbacks = {
            "FEDFUNDS": 0.045, "GDP": 20.0, "CPI": 0.03, "CORE_CPI": 0.03, "PPI": 0.03, "PCE": 0.02,
            "UNEMPLOYMENT": 0.04, "CCI": 100.0, "JOLTS": 7.0, "ISM_SERVICES": 50.0, "TREASURY_10Y": 0.04
        }
        return fallbacks.get(indicator, 0.0)

def get_macro_factors() -> Dict[str, float]:
    """Obtiene un conjunto ampliado de factores macroeconómicos con manejo de errores."""
    factors = {}
    macro_indicators = [
        "FEDFUNDS", "GDP", "CPI", "CORE_CPI", "PPI", "PCE", "UNEMPLOYMENT",
        "CCI", "JOLTS", "ISM_SERVICES", "TREASURY_10Y"
    ]
    for indicator in macro_indicators:
        factors[indicator.lower()] = get_macro_data(indicator)
    return factors

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




def get_intraday_prices(ticker, interval, start_time, end_time):
  url = f"{TRADIER_BASE_URL}/markets/timesales"
  params = {
    "symbol": ticker,
    "interval": interval,
    "start": start_time.strftime('%Y-%m-%d %H:%M:%S'),
    "end": end_time.strftime('%Y-%m-%d %H:%M:%S')
  }
  try:
    response = requests.get(url, headers=HEADERS_TRADIER, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data and 'series' in data and 'data' in data['series']:
      timesales = data['series']['data']
      timestamps = [pd.to_datetime(entry['time']) for entry in timesales]
      prices = [float(entry.get('close', 0)) for entry in timesales]
      return prices, timestamps
  except Exception as e:
    logger.error(f"Error fetching intraday prices for {ticker}: {str(e)}")
    return [], []
  return [], []





def fetch_earnings_data(start_date: str, end_date: str) -> List[Dict]:
    """Fetch earnings calendar data from FMP for a date range."""
    url = f"{FMP_BASE_URL}/earning_calendar"
    params = {"apikey": FMP_API_KEY, "from": start_date, "to": end_date}
    try:
        response = session_fmp.get(url, params=params, headers=HEADERS_FMP, timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Fetched {len(data)} earnings events from {start_date} to {end_date}")
            return data
        logger.error(f"Earnings calendar fetch failed: Status {response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Error fetching earnings data: {str(e)}")
        return []







import yfinance as yf
from scipy.stats import norm

def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculates Gamma using Black-Scholes."""
    try:
        if T <= 0:
            T = 0.0001
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except:
        return 0.0

def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Calculates Delta using Black-Scholes."""
    try:
        if T <= 0:
            T = 0.0001
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type.lower() == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    except:
        return 0.0

def calculate_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Calculates Theta (daily time decay) using Black-Scholes."""
    try:
        if T <= 0:
            T = 0.0001  # Prevent division by zero
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == "call":
            theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        daily_theta = theta / 365.0
        return min(daily_theta, 0.0)  # Ensure theta is non-positive (decay)
    except Exception as e:
        logger.error(f"Error calculating theta: {e}")
        return 0.0

@st.cache_data(ttl=CACHE_TTL)
def estimate_historical_volatility(ticker: str) -> float:
    """Estimates historical volatility using 30-day daily returns."""
    try:
        yf_ticker = "^GSPC" if ticker == "SPX" else ticker
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(period="1mo")["Close"]
        returns = hist.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        return max(0.1, min(volatility, 1.0))
    except Exception as e:
        logger.error(f"Error estimating volatility for {ticker}: {e}")
        return 0.2

@st.cache_data(ttl=CACHE_TTL)
def calculate_max_pain_strategy(ticker: str, expiry_date: datetime.date) -> float:
    """Calculates the Max Pain strike price for strategy analyzer using yfinance."""
    try:
        yf_ticker = "^GSPC" if ticker == "SPX" else ticker
        stock = yf.Ticker(yf_ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        options = stock.option_chain(expiry_str) if expiry_str in stock.options else None
        if not options:
            logger.warning(f"No option data for {ticker} on {expiry_str}")
            return current_price

        strikes = []
        pain_values = []
        for option_type in ["calls", "puts"]:
            chain = getattr(options, option_type)
            for strike in chain["strike"]:
                if strike not in strikes:
                    strikes.append(strike)

        for strike in strikes:
            total_pain = 0
            call_chain = options.calls[options.calls["strike"] == strike]
            put_chain = options.puts[options.puts["strike"] == strike]
            call_oi = call_chain["openInterest"].iloc[0] if not call_chain.empty else 0
            put_oi = put_chain["openInterest"].iloc[0] if not put_chain.empty else 0
            if current_price > strike:
                total_pain += (current_price - strike) * call_oi
            if current_price < strike:
                total_pain += (strike - current_price) * put_oi
            pain_values.append(total_pain)

        if pain_values:
            max_pain_strike = strikes[np.argmin(pain_values)]
            return max_pain_strike
        return current_price
    except Exception as e:
        logger.error(f"Error calculating Max Pain for {ticker}: {e}")
        return 0.0

@st.cache_data(ttl=CACHE_TTL)
def get_option_contracts(ticker: str, expiry_date: datetime.date) -> dict:
    """Fetches option chain and suggests contracts based on Max Pain and Gamma."""
    try:
        yf_ticker = "^GSPC" if ticker == "SPX" else ticker
        stock = yf.Ticker(yf_ticker)
        current_price = stock.history(period="1d")["Close"].iloc[-1]
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        options = stock.option_chain(expiry_str) if expiry_str in stock.options else None
        if not options:
            logger.warning(f"No option data for {ticker} on {expiry_str}")
            max_pain = current_price
            strike_step = 10.0 if current_price > 100 else 5.0
            contracts = [
                {"type": "call", "strike": round(current_price * 1.05 / strike_step) * strike_step, "gamma": 0.03, "volume": 100, "distance_to_max_pain": 0.05 * current_price},
                {"type": "call", "strike": round(current_price * 1.10 / strike_step) * strike_step, "gamma": 0.02, "volume": 80, "distance_to_max_pain": 0.10 * current_price},
                {"type": "put", "strike": round(current_price * 0.95 / strike_step) * strike_step, "gamma": 0.03, "volume": 100, "distance_to_max_pain": 0.05 * current_price},
                {"type": "put", "strike": round(current_price * 0.90 / strike_step) * strike_step, "gamma": 0.02, "volume": 80, "distance_to_max_pain": 0.10 * current_price},
                {"type": "call", "strike": round(current_price * 1.15 / strike_step) * strike_step, "gamma": 0.01, "volume": 60, "distance_to_max_pain": 0.15 * current_price}
            ]
            return {"max_pain": max_pain, "contracts": contracts}

        max_pain = calculate_max_pain_strategy(ticker, expiry_date)
        contracts = []
        sigma = estimate_historical_volatility(ticker)
        current_time = datetime.now()
        T = max((datetime.combine(expiry_date, datetime.min.time()) - current_time).total_seconds() / (365 * 24 * 60 * 60), 0.0001)
        for option_type in ["calls", "puts"]:
            chain = getattr(options, option_type)
            for _, row in chain.iterrows():
                strike = row["strike"]
                volume = row["volume"] if not pd.isna(row["volume"]) else 0
                if T <= 0:
                    T = 0.0001
                gamma = calculate_gamma(current_price, strike, T, 0.045, sigma)
                contracts.append({
                    "type": option_type,
                    "strike": strike,
                    "gamma": gamma,
                    "volume": volume,
                    "distance_to_max_pain": abs(strike - max_pain)
                })

        contracts = [c for c in contracts if c["volume"] > 0]
        contracts.sort(key=lambda x: x["gamma"], reverse=True)
        return {"max_pain": max_pain, "contracts": contracts[:5]}
    except Exception as e:
        logger.error(f"Error fetching option contracts for {ticker}: {e}")
        current_price = get_current_price(ticker)
        max_pain = current_price
        strike_step = 10.0 if current_price > 100 else 5.0
        contracts = [
            {"type": "call", "strike": round(current_price * 1.05 / strike_step) * strike_step, "gamma": 0.03, "volume": 100, "distance_to_max_pain": 0.05 * current_price},
            {"type": "call", "strike": round(current_price * 1.10 / strike_step) * strike_step, "gamma": 0.02, "volume": 80, "distance_to_max_pain": 0.10 * current_price},
            {"type": "put", "strike": round(current_price * 0.95 / strike_step) * strike_step, "gamma": 0.03, "volume": 100, "distance_to_max_pain": 0.05 * current_price},
            {"type": "put", "strike": round(current_price * 0.90 / strike_step) * strike_step, "gamma": 0.02, "volume": 80, "distance_to_max_pain": 0.10 * current_price},
            {"type": "call", "strike": round(current_price * 1.15 / strike_step) * strike_step, "gamma": 0.01, "volume": 60, "distance_to_max_pain": 0.15 * current_price}
        ]
        return {"max_pain": max_pain, "contracts": contracts}

def estimate_option_premium(S: float, K: float, T: float, r: float = 0.045, sigma: float = 0.2, option_type: str = "call") -> float:
    """Estimates the option premium using a simplified Black-Scholes model."""
    try:
        if T <= 0:
            T = 0.0001
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type.lower() == "call":
            premium = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            premium = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(premium, 0) * 100
    except:
        return 0.0

def calculate_strategy_metrics(strategy: str, spx: float, spx_current: float, call_strike: float, 
                              put_strike: float, put_strike_lower: float, call_strike_higher: float, 
                              expiry_date: datetime.date, call_strike_buy: float = 0.0, put_strike_buy: float = 0.0, 
                              ticker: str = "SPY") -> tuple:
    """Calculates the payoff, net premium, Current Delta, Decayed Delta, and Theta for the selected strategy."""
    # Log inputs for debugging
    logger.debug(f"Calculating metrics for strategy: {strategy}, ticker: {ticker}, spx: {spx}, spx_current: {spx_current}, "
                 f"call_strike: {call_strike}, put_strike: {put_strike}, put_strike_lower: {put_strike_lower}, "
                 f"call_strike_higher: {call_strike_higher}, call_strike_buy: {call_strike_buy}, put_strike_buy: {put_strike_buy}")

    try:
        expiry_datetime = datetime.combine(expiry_date, datetime.min.time())
        current_time = datetime.now()
        T = max((expiry_datetime - current_time).total_seconds() / (365 * 24 * 60 * 60), 0.0001)  # Convert to years
        sigma = estimate_historical_volatility(ticker)
        T_current = T
        T_decayed = max(T - 1/365.0, 0.0001)  # One day less for decayed delta

        if strategy == "Collar":
            if spx < put_strike_lower:
                payoff = spx - spx_current - (put_strike - put_strike_lower)
            elif spx < put_strike:
                payoff = put_strike - spx_current
            elif spx < call_strike:
                payoff = spx - spx_current
            else:
                payoff = call_strike - spx_current
            call_premium = estimate_option_premium(spx_current, call_strike, T, sigma=sigma, option_type="call")
            put_premium = estimate_option_premium(spx_current, put_strike, T, sigma=sigma, option_type="put")
            put_lower_premium = estimate_option_premium(spx_current, put_strike_lower, T, sigma=sigma, option_type="put")
            net_premium = call_premium + put_lower_premium - put_premium

            delta_current = -calculate_delta(spx_current, call_strike, T_current, 0.045, sigma, "call") + \
                            calculate_delta(spx_current, put_strike, T_current, 0.045, sigma, "put") - \
                            calculate_delta(spx_current, put_strike_lower, T_current, 0.045, sigma, "put")
            delta_decayed = -calculate_delta(spx_current, call_strike, T_decayed, 0.045, sigma, "call") + \
                            calculate_delta(spx_current, put_strike, T_decayed, 0.045, sigma, "put") - \
                            calculate_delta(spx_current, put_strike_lower, T_decayed, 0.045, sigma, "put")
            theta = -calculate_theta(spx_current, call_strike, T, 0.045, sigma, "call") + \
                    calculate_theta(spx_current, put_strike, T, 0.045, sigma, "put") - \
                    calculate_theta(spx_current, put_strike_lower, T, 0.045, sigma, "put")

        elif strategy == "Iron Condor":
            call_strike_lower = put_strike
            if spx < put_strike_lower:
                payoff = spx - put_strike_lower
            elif spx < put_strike:
                payoff = 0
            elif spx < call_strike_lower:
                payoff = 0
            elif spx < call_strike:
                payoff = call_strike_lower - spx
            else:
                payoff = call_strike_lower - call_strike
            put_lower_premium = estimate_option_premium(spx_current, put_strike_lower, T, sigma=sigma, option_type="put")
            put_premium = estimate_option_premium(spx_current, put_strike, T, sigma=sigma, option_type="put")
            call_lower_premium = estimate_option_premium(spx_current, call_strike_lower, T, sigma=sigma, option_type="call")
            call_premium = estimate_option_premium(spx_current, call_strike, T, sigma=sigma, option_type="call")
            net_premium = put_lower_premium + call_lower_premium - put_premium - call_premium

            delta_current = -calculate_delta(spx_current, put_strike, T_current, 0.045, sigma, "put") + \
                            calculate_delta(spx_current, put_strike_lower, T_current, 0.045, sigma, "put") - \
                            calculate_delta(spx_current, call_strike_lower, T_current, 0.045, sigma, "call") + \
                            calculate_delta(spx_current, call_strike, T_current, 0.045, sigma, "call")
            delta_decayed = -calculate_delta(spx_current, put_strike, T_decayed, 0.045, sigma, "put") + \
                            calculate_delta(spx_current, put_strike_lower, T_decayed, 0.045, sigma, "put") - \
                            calculate_delta(spx_current, call_strike_lower, T_decayed, 0.045, sigma, "call") + \
                            calculate_delta(spx_current, call_strike, T_decayed, 0.045, sigma, "call")
            theta = -calculate_theta(spx_current, put_strike, T, 0.045, sigma, "put") + \
                    calculate_theta(spx_current, put_strike_lower, T, 0.045, sigma, "put") - \
                    calculate_theta(spx_current, call_strike_lower, T, 0.045, sigma, "call") + \
                    calculate_theta(spx_current, call_strike, T, 0.045, sigma, "call")

        elif strategy == "Butterfly":
            mid_strike = put_strike
            wing_width = (call_strike - put_strike) / 2
            lower_strike = mid_strike - wing_width
            higher_strike = mid_strike + wing_width
            if spx < lower_strike or spx > higher_strike:
                payoff = 0
            elif spx < mid_strike:
                payoff = spx - lower_strike
            else:
                payoff = higher_strike - spx
            lower_premium = estimate_option_premium(spx_current, lower_strike, T, sigma=sigma, option_type="call")
            mid_premium = estimate_option_premium(spx_current, mid_strike, T, sigma=sigma, option_type="call") * 2
            higher_premium = estimate_option_premium(spx_current, higher_strike, T, sigma=sigma, option_type="call")
            net_premium = lower_premium + higher_premium - mid_premium

            delta_current = calculate_delta(spx_current, lower_strike, T_current, 0.045, sigma, "call") - \
                            2 * calculate_delta(spx_current, mid_strike, T_current, 0.045, sigma, "call") + \
                            calculate_delta(spx_current, higher_strike, T_current, 0.045, sigma, "call")
            delta_decayed = calculate_delta(spx_current, lower_strike, T_decayed, 0.045, sigma, "call") - \
                            2 * calculate_delta(spx_current, mid_strike, T_decayed, 0.045, sigma, "call") + \
                            calculate_delta(spx_current, higher_strike, T_decayed, 0.045, sigma, "call")
            theta = calculate_theta(spx_current, lower_strike, T, 0.045, sigma, "call") - \
                    2 * calculate_theta(spx_current, mid_strike, T, 0.045, sigma, "call") + \
                    calculate_theta(spx_current, higher_strike, T, 0.045, sigma, "call")

        elif strategy == "Call Spread":
            if spx < call_strike_buy:
                payoff = 0
            elif spx < call_strike:
                payoff = spx - call_strike_buy
            else:
                payoff = call_strike - call_strike_buy
            buy_premium = estimate_option_premium(spx_current, call_strike_buy, T, sigma=sigma, option_type="call")
            sell_premium = estimate_option_premium(spx_current, call_strike, T, sigma=sigma, option_type="call")
            net_premium = sell_premium - buy_premium

            delta_current = calculate_delta(spx_current, call_strike_buy, T_current, 0.045, sigma, "call") - \
                            calculate_delta(spx_current, call_strike, T_current, 0.045, sigma, "call")
            delta_decayed = calculate_delta(spx_current, call_strike_buy, T_decayed, 0.045, sigma, "call") - \
                            calculate_delta(spx_current, call_strike, T_decayed, 0.045, sigma, "call")
            theta = calculate_theta(spx_current, call_strike_buy, T, 0.045, sigma, "call") - \
                    calculate_theta(spx_current, call_strike, T, 0.045, sigma, "call")

        elif strategy == "Put Spread":
            if spx < put_strike_buy:
                payoff = put_strike_buy - put_strike_lower
            elif spx < put_strike:
                payoff = put_strike_buy - spx
            else:
                payoff = 0
            buy_premium = estimate_option_premium(spx_current, put_strike_buy, T, sigma=sigma, option_type="put")
            sell_premium = estimate_option_premium(spx_current, put_strike_lower, T, sigma=sigma, option_type="put")
            net_premium = sell_premium - buy_premium

            delta_current = calculate_delta(spx_current, put_strike_buy, T_current, 0.045, sigma, "put") - \
                            calculate_delta(spx_current, put_strike_lower, T_current, 0.045, sigma, "put")
            delta_decayed = calculate_delta(spx_current, put_strike_buy, T_decayed, 0.045, sigma, "put") - \
                            calculate_delta(spx_current, put_strike_lower, T_decayed, 0.045, sigma, "put")
            theta = calculate_theta(spx_current, put_strike_buy, T, 0.045, sigma, "put") - \
                    calculate_theta(spx_current, put_strike_lower, T, 0.045, sigma, "put")

        elif strategy == "Buy Call":
            if spx < call_strike_buy:
                payoff = 0
            else:
                payoff = spx - call_strike_buy
            premium = estimate_option_premium(spx_current, call_strike_buy, T, sigma=sigma, option_type="call")
            net_premium = -premium

            delta_current = calculate_delta(spx_current, call_strike_buy, T_current, 0.045, sigma, "call")
            delta_decayed = calculate_delta(spx_current, call_strike_buy, T_decayed, 0.045, sigma, "call")
            theta = calculate_theta(spx_current, call_strike_buy, T, 0.045, sigma, "call")

        elif strategy == "Buy Put":
            if spx < put_strike_buy:
                payoff = put_strike_buy - spx
            else:
                payoff = 0
            premium = estimate_option_premium(spx_current, put_strike_buy, T, sigma=sigma, option_type="put")
            net_premium = -premium

            delta_current = calculate_delta(spx_current, put_strike_buy, T_current, 0.045, sigma, "put")
            delta_decayed = calculate_delta(spx_current, put_strike_buy, T_decayed, 0.045, sigma, "put")
            theta = calculate_theta(spx_current, put_strike_buy, T, 0.045, sigma, "put")

        else:
            logger.error(f"Unhandled strategy: {strategy}")
            return 0.0, 0.0, 0.0, 0.0, 0.0  # Fallback for invalid strategy

        logger.debug(f"Strategy: {strategy}, T: {T:.6f}, Theta: {theta:.6f}")
        return payoff, net_premium, delta_current, delta_decayed, theta

    except Exception as e:
        logger.error(f"Error in calculate_strategy_metrics for {strategy}: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0  # Fallback for any errors

def suggest_strategy(max_pain: float, contracts: list, current_price: float) -> tuple[str, dict]:
    """Suggests an options strategy based on Max Pain and Gamma."""
    if not contracts or max_pain == 0.0:
        return "Collar", {
            "call_strike": current_price * 1.05,
            "put_strike": current_price * 0.95,
            "put_strike_lower": current_price * 0.9,
            "call_strike_higher": 0.0,
            "initial_price": current_price,
            "call_strike_buy": 0.0,
            "put_strike_buy": 0.0
        }

    avg_call_gamma = np.mean([c["gamma"] for c in contracts if c["type"] == "calls"]) if any(c["type"] == "calls" for c in contracts) else 0.0
    avg_put_gamma = np.mean([c["gamma"] for c in contracts if c["type"] == "puts"]) if any(c["type"] == "puts" for c in contracts) else 0.0
    call_strikes = [c["strike"] for c in contracts if c["type"] == "calls" and c["strike"] > max_pain]
    put_strikes = [c["strike"] for c in contracts if c["type"] == "puts" and c["strike"] < max_pain]

    if avg_call_gamma > 0.05 or avg_put_gamma > 0.05:
        if current_price > max_pain * 1.05:
            strategy = "Buy Call"
            strikes = {
                "call_strike": 0.0,
                "put_strike": 0.0,
                "put_strike_lower": 0.0,
                "call_strike_higher": 0.0,
                "initial_price": max_pain,
                "call_strike_buy": min(call_strikes, default=max_pain * 1.05) if call_strikes else max_pain * 1.05,
                "put_strike_buy": 0.0
            }
        elif current_price < max_pain * 0.95:
            strategy = "Buy Put"
            strikes = {
                "call_strike": 0.0,
                "put_strike": 0.0,
                "put_strike_lower": 0.0,
                "call_strike_higher": 0.0,
                "initial_price": max_pain,
                "call_strike_buy": 0.0,
                "put_strike_buy": max(put_strikes, default=max_pain * 0.95) if put_strikes else max_pain * 0.95
            }
        else:
            strategy = "Call Spread" if avg_call_gamma > avg_put_gamma else "Put Spread"
            if strategy == "Call Spread":
                strikes = {
                    "call_strike": min(call_strikes, default=max_pain * 1.1) if call_strikes else max_pain * 1.1,
                    "put_strike": 0.0,
                    "put_strike_lower": 0.0,
                    "call_strike_higher": max(call_strikes, default=max_pain * 1.2) if call_strikes else max_pain * 1.2,
                    "initial_price": max_pain,
                    "call_strike_buy": min(call_strikes, default=max_pain * 1.1) if call_strikes else max_pain * 1.1,
                    "put_strike_buy": 0.0
                }
            else:
                strikes = {
                    "call_strike": 0.0,
                    "put_strike": max(put_strikes, default=max_pain * 0.9) if put_strikes else max_pain * 0.9,
                    "put_strike_lower": min(put_strikes, default=max_pain * 0.8) if put_strikes else max_pain * 0.8,
                    "call_strike_higher": 0.0,
                    "initial_price": max_pain,
                    "call_strike_buy": 0.0,
                    "put_strike_buy": max(put_strikes, default=max_pain * 0.9) if put_strikes else max_pain * 0.9
                }
    else:
        if current_price > max_pain * 1.05 or current_price < max_pain * 0.95:
            strategy = "Iron Condor"
            strikes = {
                "call_strike": min(call_strikes, default=max_pain * 1.15) if call_strikes else max_pain * 1.15,
                "put_strike": max(put_strikes, default=max_pain * 0.95) if put_strikes else max_pain * 0.95,
                "put_strike_lower": min(put_strikes, default=max_pain * 0.85) if put_strikes else max_pain * 0.85,
                "call_strike_higher": max(call_strikes, default=max_pain * 1.25) if call_strikes else max_pain * 1.25,
                "initial_price": max_pain,
                "call_strike_buy": 0.0,
                "put_strike_buy": 0.0
            }
        else:
            strategy = "Butterfly"
            strikes = {
                "call_strike": max_pain * 1.1,
                "put_strike": max_pain,
                "put_strike_lower": max_pain * 0.9,
                "call_strike_higher": 0.0,
                "initial_price": max_pain,
                "call_strike_buy": 0.0,
                "put_strike_buy": 0.0
            }
    return strategy, strikes

def plot_strategy_payoff(strategy: str, current_price: float, current_payoff: float, price_current: float, 
                         call_strike: float, put_strike: float, put_strike_lower: float, call_strike_higher: float, 
                         expiry_date: datetime.date, call_strike_buy: float = 0.0, put_strike_buy: float = 0.0, 
                         ticker: str = "SPY") -> go.Figure:
    """Plots a dynamic payoff profile with Current Delta, Decayed Delta, and Theta curves."""
    scale = 1.0 if current_price > 1000 else 0.1
    min_price = max(10, price_current * 0.8) if scale < 1.0 else max(1000, price_current * 0.9)
    max_price = price_current * 1.2 if scale < 1.0 else price_current * 1.1
    price_range = np.linspace(min_price, max_price, 500)
    payoff = []
    delta_current = []
    delta_decayed = []
    theta = []
    for p in price_range:
        p_data, n_premium, d_current, d_decayed, t = calculate_strategy_metrics(strategy, p, price_current, call_strike, put_strike, 
                                                                               put_strike_lower, call_strike_higher, expiry_date, 
                                                                               call_strike_buy, put_strike_buy, ticker)
        payoff.append(p_data)
        delta_current.append(d_current)
        delta_decayed.append(d_decayed)
        theta.append(t)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_range, y=payoff, mode="lines", name="Payoff", line=dict(color="#39FF14", width=3),
        hovertemplate="Price: $%{x:.2f}<br>Payoff: $%{y:.2f}<br>Delta (Current): %{customdata[0]:.2f}<br>Delta (Decayed): %{customdata[1]:.2f}<br>Theta: $%{customdata[2]:.2f}",
        customdata=np.stack((delta_current, delta_decayed, theta), axis=-1)
    ))
    fig.add_trace(go.Scatter(
        x=price_range, y=delta_current, mode="lines", name="Delta (Current)", line=dict(color="#FFA500", width=2, dash="dash"),
        yaxis="y2",
        hovertemplate="Price: $%{x:.2f}<br>Delta (Current): %{y:.2f}"
    ))
    fig.add_trace(go.Scatter(
        x=price_range, y=delta_decayed, mode="lines", name="Delta (Decayed)", line=dict(color="#FF8C00", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="Price: $%{x:.2f}<br>Delta (Decayed): %{y:.2f}"
    ))
    fig.add_trace(go.Scatter(
        x=price_range, y=theta, mode="lines", name="Theta (Daily)", line=dict(color="#00CED1", width=2, dash="dot"),
        hovertemplate="Price: $%{x:.2f}<br>Theta: $%{y:.2f}"
    ))
    fig.add_trace(go.Scatter(
        x=[current_price], y=[current_payoff], mode="markers", name="Current Price",
        marker=dict(size=12, color="#FFD700", symbol="circle", line=dict(width=2, color="#FFF")),
        hovertemplate="Price: $%{x:.2f}<br>Payoff: $%{y:.2f}"
    ))
    if strategy in ["Collar", "Iron Condor", "Butterfly"]:
        if call_strike > 0:
            fig.add_hline(y=call_strike, line=dict(color="#FF4500", width=1, dash="dot"), annotation_text="Call Strike (Sell)", annotation_position="top left")
        if put_strike > 0:
            fig.add_hline(y=put_strike, line=dict(color="#32CD32", width=1, dash="dot"), annotation_text="Put Strike (Buy)", annotation_position="top left")
        if put_strike_lower > 0:
            fig.add_hline(y=put_strike_lower, line=dict(color="#800080", width=1, dash="dot"), annotation_text="Lower Put Strike (Sell)", annotation_position="top left")
        if call_strike_higher > 0:
            fig.add_hline(y=call_strike_higher, line=dict(color="#FF69B4", width=1, dash="dot"), annotation_text="Higher Call Strike (Buy)", annotation_position="top left")
    elif strategy == "Call Spread":
        if call_strike_buy > 0:
            fig.add_hline(y=call_strike_buy, line=dict(color="#FFA500", width=1, dash="dot"), annotation_text="Call Strike (Buy)", annotation_position="top left")
        if call_strike > 0:
            fig.add_hline(y=call_strike, line=dict(color="#FF4500", width=1, dash="dot"), annotation_text="Call Strike (Sell)", annotation_position="top left")
    elif strategy == "Put Spread":
        if put_strike_buy > 0:
            fig.add_hline(y=put_strike_buy, line=dict(color="#00CED1", width=1, dash="dot"), annotation_text="Put Strike (Buy)", annotation_position="top left")
        if put_strike_lower > 0:
            fig.add_hline(y=put_strike_lower, line=dict(color="#800080", width=1, dash="dot"), annotation_text="Lower Put Strike (Sell)", annotation_position="top left")
    elif strategy == "Buy Call" and call_strike_buy > 0:
        fig.add_hline(y=call_strike_buy, line=dict(color="#FFA500", width=1, dash="dot"), annotation_text="Call Strike (Buy)", annotation_position="top left")
    elif strategy == "Buy Put" and put_strike_buy > 0:
        fig.add_hline(y=put_strike_buy, line=dict(color="#00CED1", width=1, dash="dot"), annotation_text="Put Strike (Buy)", annotation_position="top left")
    
    fig.add_hline(y=price_current, line=dict(color="#666", width=1, dash="dash"), annotation_text="Initial Price", annotation_position="top left")
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    fig.update_layout(
        title=f"{ticker} {strategy} Payoff Profile",
        xaxis_title="Price ($)",
        yaxis_title="Payoff/Theta ($)",
        yaxis2=dict(title="Delta", overlaying="y", side="right", showgrid=False),
        template="plotly_dark",
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        xaxis=dict(gridcolor="#333", tickformat=",.2f", showgrid=True),
        yaxis=dict(gridcolor="#333", tickformat=",.2f", showgrid=True),
        plot_bgcolor="#000",
        paper_bgcolor="#000",
        font=dict(color="#FFF", family="Arial, sans-serif"),
        hovermode="x unified",
        updatemenus=[
            dict(
                buttons=[
                    dict(label="Zoom In", method="update", args=[{"xaxis.range": [current_price * 0.9, current_price * 1.1]}]),
                    dict(label="Zoom Out", method="update", args=[{"xaxis.range": [min_price, max_price]}]),
                ],
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    return fig








# --- Main App --
# --- Main App ---
# --- Main App ---
# --- Main App ---
# --- Main App ---
# --- Main App ---
def main():
    # Pantalla de autenticación sin logo
    initialize_passwords_db()
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        # Estilo profesional y centrado para el login
        st.markdown("""
        <style>
        /* Fondo global negro puro */
        .stApp {
            background-color: #000000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh; /* Centra verticalmente en toda la pantalla */
        }
        .login-container {
            background: #1E1E1E; /* Gris oscuro para contraste sutil */
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.3); /* Sombra azul neón */
            width: 100%;
            max-width: 400px; /* Ancho fijo para consistencia */
            text-align: center;
            border: 1px solid rgba(0, 255, 255, 0.2); /* Borde azul sutil */
        }
        .login-title {
            font-size: 28px;
            font-weight: 700;
            color: #00FFFF; /* Azul neón para el título */
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
            margin-bottom: 20px;
            letter-spacing: 1px;
        }
        .login-input {
            background-color: #2D2D2D;
            color: #FFFFFF;
            border: 1px solid rgba(57, 255, 20, 0.3); /* Borde verde lima sutil */
            border-radius: 5px;
            padding: 10px;
            width: 100%;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .login-button {
            background: linear-gradient(90deg, #39FF14, #00FFFF); /* Degradado verde a azul */
            color: #1E1E1E;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }
        .login-button:hover {
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.8);
            transform: scale(1.05);
        }
        .loading-container {
            text-align: center;
            padding: 25px;
            background: #1E1E1E;
            border-radius: 12px;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
            border: 1px solid rgba(0, 255, 255, 0.2);
            margin-top: 20px;
        }
        .loading-text {
            font-size: 24px;
            font-weight: 600;
            color: #39FF14; /* Verde lima */
            text-shadow: 0 0 10px rgba(57, 255, 20, 0.8);
            letter-spacing: 1px;
        }
        .spinner-pro {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255, 255, 255, 0.2);
            border-top: 4px solid #00FFFF; /* Azul neón */
            border-radius: 50%;
            animation: spin-pro 1s ease-in-out infinite;
            margin: 15px auto 0;
        }
        @keyframes spin-pro {
            0% { transform: rotate(0deg); }
            50% { transform: rotate(180deg); border-top-color: #39FF14; }
            100% { transform: rotate(360deg); border-top-color: #00FFFF; }
        }
        </style>
        """, unsafe_allow_html=True)

        # Contenedor centrado para el login
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔒 VIP ACCESS</div>', unsafe_allow_html=True)
        password = st.text_input("Enter your password", type="password", key="login_input", help="Enter your VIP password")
        if st.button("LogIn", key="login_button"):
            if not password:
                st.error("❌ Please enter a password.")
            elif authenticate_password(password):
                st.session_state["authenticated"] = True
                with st.empty():
                    st.markdown("""
                    <div class="loading-container">
                        <div class="loading-text">✅ ACCESS GRANTED</div>
                        <div class="spinner-pro"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # Solo una columna para el título, sin logo
    st.markdown("""
        <div class="header-container">
            <div class="header-title">ℙℝ𝕆 𝔼𝕊ℂ𝔸ℕℕ𝔼ℝ®</div>
        </div>
    """, unsafe_allow_html=True)

    # Estilos personalizados con tabs y botones de descarga ultra compactos y futuristas
    st.markdown("""
        <style>
        /* Fondo global negro puro como las gráficas */
        .stApp {
            background-color: #000000;
        }
        .stTextInput, .stSelectbox {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }
        .stSpinner > div > div {
            border-color: #32CD32 !important;
        }
        /* Menú sin rectángulo */
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            background: none; /* Sin fondo rectangular */
            padding: 5px;
            gap: 2px;
            margin-top: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 5px 10px;
            margin: 2px;
            color: rgba(57, 255, 20, 0.7); /* Verde lima apagado como base */
            background: #000000; /* Negro puro para combinar con el fondo */
            border: 1px solid rgba(57, 255, 20, 0.15); /* Borde sutil de neón, 50% menos brillante */
            border-radius: 5px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 0 2.5px rgba(57, 255, 20, 0.1); /* Brillo reducido al 50% */
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #39FF14; /* Verde lima brillante al pasar el ratón */
            color: #1E1E1E;
            transform: translateY(-2px);
            box-shadow: 0 4px 5px rgba(57, 255, 20, 0.4); /* Brillo reducido al 50% */
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #00FFFF; /* Azul neón para tab activo */
            color: #1E1E1E;
            font-weight: 700;
            transform: scale(1.1); /* Se "infla" un poco más */
            box-shadow: 0 0 7.5px rgba(0, 255, 255, 0.4); /* Brillo reducido al 50% */
            border: 1px solid rgba(0, 255, 255, 0.5); /* Borde menos brillante */
        }
        /* Estilo para botones de descarga */
        .stDownloadButton > button {
            padding: 5px 10px;
            margin: 2px;
            color: rgba(57, 255, 20, 0.7); /* Verde lima apagado como base */
            background: #000000; /* Negro puro para combinar con el fondo */
            border: 1px solid rgba(57, 255, 20, 0.15); /* Borde sutil de neón, 50% menos brillante */
            border-radius: 5px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 0 2.5px rgba(57, 255, 20, 0.1); /* Brillo reducido al 50% */
        }
        .stDownloadButton > button:hover {
            background: #39FF14; /* Verde lima brillante al pasar el ratón */
            color: #1E1E1E;
            transform: translateY(-2px);
            box-shadow: 0 4px 5px rgba(57, 255, 20, 0.4); /* Brillo reducido al 50% */
        }
        </style>
    """, unsafe_allow_html=True)

    # Definición de los tabs (Tab 13 removed)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab11, tab12 = st.tabs([
        "Gummy Data Bubbles® |", "Market Scanner |", "News |", "Institutional Holders |",
        "Options Order Flow |", "Analyst Rating Flow |", "Elliott Pulse® |", "Crypto Insights |",
        "Projection |", "Performance Map |"
    ])

    # Tab 1: Gummy Data Bubbles®
    with tab1:
        ticker = st.text_input("Ticker", value="SPY", key="ticker_input").upper()
        
        expiration_dates = get_expiration_dates(ticker)
        if not expiration_dates:
            st.error(f"What were you thinking, '{ticker}'? You're a trader and you mess this up? If you trade like this, you're doomed!")
            return
        
        # Ensure we select a valid expiration date (not in the future)
        expiration_date = st.selectbox("Expiration Date", expiration_dates, key="expiration_date")
        selected_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        current_date = datetime.now()
        if selected_date > current_date:
            
            valid_dates = [d for d in expiration_dates if datetime.strptime(d, '%Y-%m-%d') <= current_date]
            expiration_date = max(valid_dates, default=expiration_date) if valid_dates else expiration_date
        
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
                st.warning(f"No options data available for {ticker} on {expiration_date}. Using default values.")
                options_data = [
                    {"strike": current_price - 10, "option_type": "call", "volume": 1000, "implied_volatility": 0.2, "greeks": {"gamma": 0.01}, "bid": 1.0, "ask": 1.2, "open_interest": 500},
                    {"strike": current_price + 10, "option_type": "put", "volume": 1200, "implied_volatility": 0.22, "greeks": {"gamma": 0.015}, "bid": 1.1, "ask": 1.3, "open_interest": 600}
                ]
            
            # Procesamiento vectorizado corregido, incluyendo Gamma
            strikes = np.array([float(opt.get("strike", 0)) for opt in options_data if opt and isinstance(opt, dict)])
            option_types = np.array([opt.get("option_type", "").upper() for opt in options_data if opt and isinstance(opt, dict)])
            ois = np.array([int(opt.get("open_interest", 0)) for opt in options_data if opt and isinstance(opt, dict)])
            volumes = np.array([int(opt.get("volume", 0)) for opt in options_data if opt and isinstance(opt, dict)])
            gammas = np.array([float(opt.get("greeks", {}).get("gamma", 0)) if isinstance(opt.get("greeks", {}), dict) else 0
                            for opt in options_data if opt and isinstance(opt, dict)])
            ivs = np.array([float(opt.get("implied_volatility", 0)) if opt.get("implied_volatility") else 0
                            for opt in options_data if opt and isinstance(opt, dict)])
            
            processed_data = {}
            unique_strikes = np.unique(strikes)
            for strike in unique_strikes:
                mask = strikes == strike
                processed_data[strike] = {
                    "CALL": {
                        "OI": np.sum(ois[mask & (option_types == "CALL")]),
                        "Volume": np.sum(volumes[mask & (option_types == "CALL")]),
                        "Gamma": np.sum(gammas[mask & (option_types == "CALL")]),
                        "IV": np.mean(ivs[mask & (option_types == "CALL")])
                    },
                    "PUT": {
                        "OI": np.sum(ois[mask & (option_types == "PUT")]),
                        "Volume": np.sum(volumes[mask & (option_types == "PUT")]),
                        "Gamma": np.sum(gammas[mask & (option_types == "PUT")]),
                        "IV": np.mean(ivs[mask & (option_types == "PUT")])
                    }
                }
            
            if not processed_data:
                st.error("No valid data to display.")
                return
            
            prices, _ = get_historical_prices_combined(ticker)
            historical_prices = prices
            touched_strikes = detect_touched_strikes(processed_data.keys(), historical_prices)
            max_pain = calculate_max_pain_optimized(options_data)
            df = analyze_contracts(ticker, expiration_date, current_price, options_data=options_data)
            if df.empty:
                # Construct a fallback DataFrame from options_data
                df = pd.DataFrame(options_data)[["strike", "option_type", "open_interest", "volume", "bid"]]
                if df.empty:
                    df = pd.DataFrame(columns=["strike", "option_type", "open_interest", "volume", "bid"])
            max_pain_strike, max_pain_df = calculate_max_pain(df)
            if max_pain_strike is None:
                max_pain_strike = current_price  # Fallback to current price
            
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
            
            st.write(f"Current Price of {ticker}: ${current_price:.2f} (Updated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')})")
            st.write(f"**Max Pain Strike (Optimized):** ${max_pain if max_pain else 'N/A'}")
            
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
            
            # Nueva sección NET_FLOW_REALTIME_ELITE_ optimizada para movimientos del MM
            st.markdown('<div class="main-title">> NET_FLOW</div>', unsafe_allow_html=True)
            
            # Configuración de actualización automática cada 220 segundos
            st_autorefresh(interval=220 * 1000, key="netflowrefresh")

            # Usar el mismo ticker y expiration_date de arriba
            st.write(f"**Using Ticker:** {ticker} | **Expiration Date:** {expiration_date}")

            # Usar una fecha histórica para evitar problemas con datos futuros
            today = datetime(2025, 4, 16)  # Usamos una fecha histórica para datos más realistas
            start_time = today.replace(hour=5, minute=0)  # 5:00 AM
            end_time = today.replace(hour=20, minute=0)   # 8:00 PM
            interval = "1min"

            # Obtener datos intradía (precios y volumen)
            try:
                prices, timestamps = get_intraday_prices(ticker, interval, start_time=start_time, end_time=end_time)
                if not prices or len(prices) < 2:
                    raise ValueError("No intraday prices returned.")
            except Exception as e:
                st.error(f"Unable to fetch intraday prices for {ticker} from 5 AM to 8 PM. Using fallback data. Error: {str(e)}")
                # Simular precios realistas para SPY (basado en un rango típico)
                time_bins = pd.date_range(start=start_time, end=end_time, freq='H')
                prices = [current_price * (1 + 0.005 * np.sin(i)) for i in range(len(time_bins))]  # Simulación de precios
                timestamps = time_bins.tolist()

            # Obtener volumen intradía del subyacente
            timesales_data = []
            try:
                url = f"{TRADIER_BASE_URL}/markets/timesales"
                params = {
                    "symbol": ticker,
                    "interval": interval,
                    "start": start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    "end": end_time.strftime('%Y-%m-%d %H:%M:%S')
                }
                response = requests.get(url, headers=HEADERS_TRADIER, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data and 'series' in data and 'data' in data['series']:
                    timesales_data = data['series']['data']
                else:
                    raise ValueError("No timesales data returned.")
            except Exception as e:
                st.error(f"Unable to fetch timesales data for {ticker} from 5 AM to 8 PM. Using fallback volume. Error: {str(e)}")
                timesales_data = [{"time": t.strftime('%Y-%m-%d %H:%M:%S'), "volume": 1000 * (1 + 0.3 * np.sin(i))} 
                                for i, t in enumerate(pd.date_range(start=start_time, end=end_time, freq='H'))]

            # Agrupar datos por hora
            time_bins = pd.date_range(start=start_time, end=end_time, freq='H')
            prices_dict = {pd.to_datetime(ts): price for ts, price in zip(timestamps, prices)}

            # Mapear timestamps a volúmenes del subyacente
            underlying_volumes = {}
            for entry in timesales_data:
                ts = pd.to_datetime(entry['time'])
                volume = entry.get('volume', 0)
                underlying_volumes[ts] = volume

            # Procesar datos para la gráfica y cálculos avanzados
            mm_data = {
                "time": [],
                "net_drift": [],           # Net Drift ($M)
                "underlying_price": [],    # Precio subyacente
                "calls_oi_pressure": [],   # Presión de OI de Calls
                "puts_oi_pressure": [],    # Presión de OI de Puts
                "gamma_exposure": [],      # Gamma Exposure
                "max_pain_distance": [],   # Distancia al Max Pain
                "volatility_skew": [],     # Volatility Skew
                "momentum": [],            # Momentum del precio
                "mm_score": [],            # Puntaje de dirección del MM
                "mm_direction": []         # Dirección predicha del MM
            }
            previous_price = None
            max_volume = max(underlying_volumes.values(), default=1)
            max_net_drift = 0
            max_price = 0
            max_calls_oi = 0
            max_puts_oi = 0
            max_gamma = 0
            max_skew = 0
            max_momentum = 0
            max_mm_score = 0

            # Calcular OI y Gamma por strike para identificar niveles clave
            strikes_data = {}
            for opt in options_data:
                strike = float(opt.get("strike", 0))
                opt_type = opt.get("option_type", "").upper()
                oi = int(opt.get("open_interest", 0))
                gamma = float(opt.get("greeks", {}).get("gamma", 0) or 0)
                if strike not in strikes_data:
                    strikes_data[strike] = {"CALL": {"OI": 0, "Gamma": 0}, "PUT": {"OI": 0, "Gamma": 0}}
                strikes_data[strike][opt_type]["OI"] += oi
                strikes_data[strike][opt_type]["Gamma"] += gamma

            # Identificar el Gamma Wall (strike con mayor Gamma Exposure)
            gamma_by_strike = {strike: (data["CALL"]["Gamma"] + data["PUT"]["Gamma"]) * (current_price / 100) 
                            for strike, data in strikes_data.items()}
            gamma_wall = max(gamma_by_strike.items(), key=lambda x: abs(x[1]), default=(current_price, 0))[0]

            for t in time_bins:
                start_hour = t
                end_hour = t + pd.Timedelta(hours=1)
                hourly_timestamps = [ts for ts in timestamps if start_hour <= pd.to_datetime(ts) < end_hour]
                if not hourly_timestamps:
                    continue

                # Promedio del precio subyacente
                hourly_prices = [prices_dict.get(pd.to_datetime(ts), current_price) for ts in hourly_timestamps]
                avg_price = np.mean(hourly_prices) if hourly_prices else current_price

                # Calcular primas, OI, IV y Gamma de Calls y Puts
                calls_premium = 0
                puts_premium = 0
                calls_oi = 0
                puts_oi = 0
                iv_calls_list = []
                iv_puts_list = []
                gamma_calls = 0
                gamma_puts = 0

                for opt in options_data:
                    opt_type = opt.get("option_type", "").lower()
                    bid = opt.get("bid", 0.0)
                    ask = opt.get("ask", 0.0)
                    oi = int(opt.get("open_interest", 0) or 0)
                    iv = float(opt.get("implied_volatility", 0) or 0)
                    gamma = float(opt.get("greeks", {}).get("gamma", 0) or 0)

                    if bid <= 0 or ask <= 0:
                        logger.warning(f"Non-positive bid/ask for {ticker} option: bid={bid}, ask={ask}")
                        continue

                    premium = (float(bid) + float(ask)) / 2 * oi * 100
                    if opt_type == "call":
                        calls_premium += premium
                        calls_oi += oi
                        if iv > 0:
                            iv_calls_list.append(iv)
                        gamma_calls += gamma
                    elif opt_type == "put":
                        puts_premium += premium
                        puts_oi += oi
                        if iv > 0:
                            iv_puts_list.append(iv)
                        gamma_puts += gamma

                # Calcular métricas clave
                net_drift = calls_premium - puts_premium
                net_drift_m = net_drift / 1_000_000 if calls_premium + puts_premium > 0 else 0
                hourly_underlying_vol = sum(vol for ts, vol in underlying_volumes.items() if start_hour <= ts < end_hour) or 1000

                # Calcular Volatility Skew
                iv_calls_avg = np.mean(iv_calls_list) * 100 if iv_calls_list else 0.0
                iv_puts_avg = np.mean(iv_puts_list) * 100 if iv_puts_list else 0.0
                volatility_skew = (iv_calls_avg - iv_puts_avg) * (calls_oi / (puts_oi + 1)) if puts_oi > 0 else 0

                # Calcular Gamma Exposure total
                total_gamma = (gamma_calls + gamma_puts) * (current_price / 100)

                # Calcular presión de OI por encima y por debajo del precio actual
                calls_oi_above = sum(data["CALL"]["OI"] for strike, data in strikes_data.items() if strike > avg_price)
                puts_oi_below = sum(data["PUT"]["OI"] for strike, data in strikes_data.items() if strike < avg_price)
                calls_oi_pressure = calls_oi_above / (max(1, abs(avg_price - gamma_wall) + 1)) if gamma_wall != avg_price else 0
                puts_oi_pressure = puts_oi_below / (max(1, abs(avg_price - gamma_wall) + 1)) if gamma_wall != avg_price else 0

                # Calcular Momentum
                momentum = 0
                if previous_price is not None:
                    price_change = (avg_price - previous_price) / previous_price if previous_price != 0 else 0
                    momentum = price_change * (hourly_underlying_vol / max_volume)
                previous_price = avg_price

                # Calcular la distancia al Max Pain y la presión hacia él
                max_pain_distance = (avg_price - max_pain_strike) / max_pain_strike if max_pain_strike != 0 else 0
                max_pain_pressure = -max_pain_distance * (calls_oi + puts_oi) / 1000  # Negativo para indicar dirección hacia Max Pain

                # Calcular el puntaje del MM (MM Score)
                # Factores: Max Pain Pressure (40%), Net Drift (20%), Gamma Exposure (20%), OI Pressure (10%), Momentum (10%)
                mm_score = (
                    (max_pain_pressure * 0.4) +
                    (-net_drift_m * 0.2) +  # Negativo porque el MM tiende a ir contra el flujo
                    (total_gamma * 0.2) +
                    ((calls_oi_pressure - puts_oi_pressure) * 0.1) +
                    (-momentum * 0.1)  # Negativo para contrarrestar el momentum
                )

                # Determinar la dirección predicha del MM
                mm_direction = "Down" if mm_score < 0 else "Up" if mm_score > 0 else "Neutral"

                # Guardar datos
                mm_data["time"].append(t)
                mm_data["net_drift"].append(net_drift_m)
                mm_data["underlying_price"].append(avg_price)
                mm_data["calls_oi_pressure"].append(calls_oi_pressure)
                mm_data["puts_oi_pressure"].append(puts_oi_pressure)
                mm_data["gamma_exposure"].append(total_gamma)
                mm_data["max_pain_distance"].append(max_pain_distance)
                mm_data["volatility_skew"].append(volatility_skew)
                mm_data["momentum"].append(momentum)
                mm_data["mm_score"].append(mm_score)
                mm_data["mm_direction"].append(mm_direction)

                # Actualizar máximos para normalización
                max_net_drift = max(max_net_drift, abs(net_drift_m))
                max_price = max(max_price, avg_price)
                max_calls_oi = max(max_calls_oi, calls_oi_pressure)
                max_puts_oi = max(max_puts_oi, puts_oi_pressure)
                max_gamma = max(max_gamma, total_gamma)
                max_skew = max(max_skew, abs(volatility_skew))
                max_momentum = max(max_momentum, abs(momentum))
                max_mm_score = max(max_mm_score, abs(mm_score))

            # Calcular medias móviles para suavizar las líneas (ventana de 3 horas)
            window = 3
            mm_df = pd.DataFrame(mm_data)
            mm_df["net_drift_ma"] = mm_df["net_drift"].rolling(window=window, min_periods=1).mean()
            mm_df["underlying_price_ma"] = mm_df["underlying_price"].rolling(window=window, min_periods=1).mean()
            mm_df["calls_oi_pressure_ma"] = mm_df["calls_oi_pressure"].rolling(window=window, min_periods=1).mean()
            mm_df["puts_oi_pressure_ma"] = mm_df["puts_oi_pressure"].rolling(window=window, min_periods=1).mean()
            mm_df["gamma_exposure_ma"] = mm_df["gamma_exposure"].rolling(window=window, min_periods=1).mean()

            # Normalizar valores para la visualización
            net_drift_normalized = [v / (max_net_drift + 1e-10) for v in mm_df["net_drift_ma"]]
            price_normalized = [(v - min(mm_df["underlying_price"])) / (max_price - min(mm_df["underlying_price"]) + 1e-10) 
                                for v in mm_df["underlying_price_ma"]]
            calls_oi_normalized = [v / (max_calls_oi + 1e-10) for v in mm_df["calls_oi_pressure_ma"]]
            puts_oi_normalized = [v / (max_puts_oi + 1e-10) for v in mm_df["puts_oi_pressure_ma"]]
            gamma_normalized = [v / (max_gamma + 1e-10) for v in mm_df["gamma_exposure_ma"]]

            # Crear gráfica de líneas
            fig = go.Figure()

            # Línea de Net Drift (verde)
            fig.add_trace(go.Scatter(
                x=mm_df["time"],
                y=net_drift_normalized,
                mode='lines',
                line=dict(color="#00FF00", width=2),
                name="Net Drift",
                hovertemplate=(
                    "Time: %{x|%I:%M %p}<br>" +
                    "Net Drift: $%{customdata[0]:.2f}M<br>" +
                    "Price: $%{customdata[1]:.2f}<br>" +
                    "Calls OI Press: %{customdata[2]:,.0f}<br>" +
                    "Puts OI Press: %{customdata[3]:,.0f}<br>" +
                    "Gamma Exp: %{customdata[4]:.2f}<br>" +
                    "Max Pain Dist: %{customdata[5]:.2%}<br>" +
                    "Vol Skew: %{customdata[6]:.2f}<br>" +
                    "Momentum: %{customdata[7]:.2f}<br>" +
                    "MM Score: %{customdata[8]:.2f}<br>" +
                    "MM Dir: %{customdata[9]}"
                ),
                customdata=list(zip(
                    mm_df["net_drift"],
                    mm_df["underlying_price"],
                    mm_df["calls_oi_pressure"],
                    mm_df["puts_oi_pressure"],
                    mm_df["gamma_exposure"],
                    mm_df["max_pain_distance"],
                    mm_df["volatility_skew"],
                    mm_df["momentum"],
                    mm_df["mm_score"],
                    mm_df["mm_direction"]
                ))
            ))

            # Línea de Precio Subyacente (azul)
            fig.add_trace(go.Scatter(
                x=mm_df["time"],
                y=price_normalized,
                mode='lines',
                line=dict(color="#00FFFF", width=2),
                name="Underlying Price",
                hovertemplate=(
                    "Time: %{x|%I:%M %p}<br>" +
                    "Net Drift: $%{customdata[0]:.2f}M<br>" +
                    "Price: $%{customdata[1]:.2f}<br>" +
                    "Calls OI Press: %{customdata[2]:,.0f}<br>" +
                    "Puts OI Press: %{customdata[3]:,.0f}<br>" +
                    "Gamma Exp: %{customdata[4]:.2f}<br>" +
                    "Max Pain Dist: %{customdata[5]:.2%}<br>" +
                    "Vol Skew: %{customdata[6]:.2f}<br>" +
                    "Momentum: %{customdata[7]:.2f}<br>" +
                    "MM Score: %{customdata[8]:.2f}<br>" +
                    "MM Dir: %{customdata[9]}"
                ),
                customdata=list(zip(
                    mm_df["net_drift"],
                    mm_df["underlying_price"],
                    mm_df["calls_oi_pressure"],
                    mm_df["puts_oi_pressure"],
                    mm_df["gamma_exposure"],
                    mm_df["max_pain_distance"],
                    mm_df["volatility_skew"],
                    mm_df["momentum"],
                    mm_df["mm_score"],
                    mm_df["mm_direction"]
                ))
            ))

            # Línea de Calls OI Pressure (amarillo)
            fig.add_trace(go.Scatter(
                x=mm_df["time"],
                y=calls_oi_normalized,
                mode='lines',
                line=dict(color="#FFD700", width=2),
                name="Calls OI Pressure",
                hovertemplate=(
                    "Time: %{x|%I:%M %p}<br>" +
                    "Net Drift: $%{customdata[0]:.2f}M<br>" +
                    "Price: $%{customdata[1]:.2f}<br>" +
                    "Calls OI Press: %{customdata[2]:,.0f}<br>" +
                    "Puts OI Press: %{customdata[3]:,.0f}<br>" +
                    "Gamma Exp: %{customdata[4]:.2f}<br>" +
                    "Max Pain Dist: %{customdata[5]:.2%}<br>" +
                    "Vol Skew: %{customdata[6]:.2f}<br>" +
                    "Momentum: %{customdata[7]:.2f}<br>" +
                    "MM Score: %{customdata[8]:.2f}<br>" +
                    "MM Dir: %{customdata[9]}"
                ),
                customdata=list(zip(
                    mm_df["net_drift"],
                    mm_df["underlying_price"],
                    mm_df["calls_oi_pressure"],
                    mm_df["puts_oi_pressure"],
                    mm_df["gamma_exposure"],
                    mm_df["max_pain_distance"],
                    mm_df["volatility_skew"],
                    mm_df["momentum"],
                    mm_df["mm_score"],
                    mm_df["mm_direction"]
                ))
            ))

            # Línea de Puts OI Pressure (rojo)
            fig.add_trace(go.Scatter(
                x=mm_df["time"],
                y=puts_oi_normalized,
                mode='lines',
                line=dict(color="#FF4500", width=2),
                name="Puts OI Pressure",
                hovertemplate=(
                    "Time: %{x|%I:%M %p}<br>" +
                    "Net Drift: $%{customdata[0]:.2f}M<br>" +
                    "Price: $%{customdata[1]:.2f}<br>" +
                    "Calls OI Press: %{customdata[2]:,.0f}<br>" +
                    "Puts OI Press: %{customdata[3]:,.0f}<br>" +
                    "Gamma Exp: %{customdata[4]:.2f}<br>" +
                    "Max Pain Dist: %{customdata[5]:.2%}<br>" +
                    "Vol Skew: %{customdata[6]:.2f}<br>" +
                    "Momentum: %{customdata[7]:.2f}<br>" +
                    "MM Score: %{customdata[8]:.2f}<br>" +
                    "MM Dir: %{customdata[9]}"
                ),
                customdata=list(zip(
                    mm_df["net_drift"],
                    mm_df["underlying_price"],
                    mm_df["calls_oi_pressure"],
                    mm_df["puts_oi_pressure"],
                    mm_df["gamma_exposure"],
                    mm_df["max_pain_distance"],
                    mm_df["volatility_skew"],
                    mm_df["momentum"],
                    mm_df["mm_score"],
                    mm_df["mm_direction"]
                ))
            ))

            # Línea de Gamma Exposure (púrpura)
            fig.add_trace(go.Scatter(
                x=mm_df["time"],
                y=gamma_normalized,
                mode='lines',
                line=dict(color="#FF00FF", width=2),
                name="Gamma Exposure",
                hovertemplate=(
                    "Time: %{x|%I:%M %p}<br>" +
                    "Net Drift: $%{customdata[0]:.2f}M<br>" +
                    "Price: $%{customdata[1]:.2f}<br>" +
                    "Calls OI Press: %{customdata[2]:,.0f}<br>" +
                    "Puts OI Press: %{customdata[3]:,.0f}<br>" +
                    "Gamma Exp: %{customdata[4]:.2f}<br>" +
                    "Max Pain Dist: %{customdata[5]:.2%}<br>" +
                    "Vol Skew: %{customdata[6]:.2f}<br>" +
                    "Momentum: %{customdata[7]:.2f}<br>" +
                    "MM Score: %{customdata[8]:.2f}<br>" +
                    "MM Dir: %{customdata[9]}"
                ),
                customdata=list(zip(
                    mm_df["net_drift"],
                    mm_df["underlying_price"],
                    mm_df["calls_oi_pressure"],
                    mm_df["puts_oi_pressure"],
                    mm_df["gamma_exposure"],
                    mm_df["max_pain_distance"],
                    mm_df["volatility_skew"],
                    mm_df["momentum"],
                    mm_df["mm_score"],
                    mm_df["mm_direction"]
                ))
            ))

            # Líneas horizontales para Max Pain y Gamma Wall
            max_pain_norm = (max_pain_strike - min(mm_df["underlying_price"])) / (max_price - min(mm_df["underlying_price"]) + 1e-10)
            gamma_wall_norm = (gamma_wall - min(mm_df["underlying_price"])) / (max_price - min(mm_df["underlying_price"]) + 1e-10)

            fig.add_hline(
                y=max_pain_norm,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"Max Pain: ${max_pain_strike:.2f}",
                annotation_position="top",
                annotation_font_color="yellow"
            )

            fig.add_hline(
                y=gamma_wall_norm,
                line_dash="dash",
                line_color="purple",
                annotation_text=f"Gamma Wall: ${gamma_wall:.2f}",
                annotation_position="top right",
                annotation_font_color="purple"
            )

            # Agregar anotación con predicción del MM
            latest_mm_score = mm_df["mm_score"].iloc[-1] if not mm_df.empty else 0
            latest_mm_direction = mm_df["mm_direction"].iloc[-1] if not mm_df.empty else "Neutral"
            speculation = (
                f"MM Likely to Push: {latest_mm_direction}<br>" +
                f"Max Pain Pressure: {max_pain_pressure:.2f}<br>" +
                f"Gamma Wall Influence: {gamma_wall:.2f}<br>" +
                f"Net OI Pressure: {(calls_oi_pressure - puts_oi_pressure):.2f}"
            )
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=speculation,
                showarrow=False,
                font=dict(color="#FFD700", size=12),
                bgcolor="rgba(0, 0, 0, 0.5)",
                bordercolor="#39FF14",
                borderwidth=1,
                borderpad=4
            )

            # Configurar diseño
            fig.update_layout(
                title=f"MM Flow Elite - {ticker} | {datetime.now().strftime('%Y-%m-%d')}",
                xaxis=dict(
                    title="Time",
                    tickformat="%I:%M %p",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    range=[start_time, end_time]
                ),
                yaxis=dict(
                    title="Normalized Value",
                    showgrid=True,
                    gridcolor="rgba(255, 255, 255, 0.1)",
                    range=[-0.1, 1.1]
                ),
                template="plotly_dark",
                plot_bgcolor="#0A0A0A",
                paper_bgcolor="#0A0A0A",
                font=dict(color="#FFFFFF", family="Courier New, monospace"),
                hovermode="x unified",
                height=600,
                margin=dict(l=80, r=80, t=80, b=60),
                legend=dict(
                    yanchor="top",
                    y=1.15,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(0, 0, 0, 0.5)",
                    bordercolor="#39FF14",
                    borderwidth=1
                )
            )

            st.plotly_chart(fig)

            # Pie de página
            st.markdown(f'<div class="footer-text">> LAST_UPDATED: {datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")} | POWERED_BY_OZY_ANALYTICS_</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("*Developed by Ozy | © 2025*")



    
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
        
        # Inicializar st.session_state para latest_news si no existe
        if "latest_news" not in st.session_state:
            with st.spinner("Fetching initial market news..."):
                google_news = fetch_google_news(["SPY"])  # SPY como default para mercado general
                bing_news = fetch_bing_news(["SPY"])
                st.session_state["latest_news"] = google_news + bing_news if google_news or bing_news else None
        
        # Sección de noticias
        st.markdown("#### Search News")
        keywords = st.text_input("Enter keywords (comma-separated):", "Trump", key="news_keywords").split(",")
        keywords = [k.strip() for k in keywords if k.strip()]
        
        if st.button("Fetch News", key="fetch_news"):
            with st.spinner("Fetching news..."):
                google_news = fetch_google_news(keywords)
                bing_news = fetch_bing_news(keywords)
                latest_news = google_news + bing_news
                
                if latest_news:
                    st.session_state["latest_news"] = latest_news
                    for idx, article in enumerate(latest_news[:10], 1):
                        st.markdown(f"### {idx}. [{article['title']}]({article['link']})")
                        st.markdown(f"**Published:** {article['time']}\n")
                        st.markdown("---")
                else:
                    st.error("No recent news found.")
                    st.session_state["latest_news"] = None
        
        # Separador
        st.markdown("---")
        
        # Sección de sentimientos (siempre visible)
        st.markdown("#### Market Sentiment (Based on Latest News)")
        if st.session_state["latest_news"]:
            sentiment_score, sentiment_text = calculate_retail_sentiment(st.session_state["latest_news"])
            volatility_score, volatility_text = calculate_volatility_sentiment(st.session_state["latest_news"])
            
            # Dividir en columnas para Retail y Volatility Sentiment
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Retail Sentiment")
                fig_sentiment = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=sentiment_score * 100,
                    delta={'reference': 50, 'relative': True, 'valueformat': '.2%'},
                    title={'text': "Retail Sentiment Score", 'font': {'size': 16, 'color': '#FFFFFF'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#FFFFFF", 'tickfont': {'color': "#FFFFFF"}},
                        'bar': {'color': "#32CD32" if sentiment_score > 0.5 else "#FF4500", 'thickness': 0.2},
                        'bgcolor': "rgba(0, 0, 0, 0.1)",
                        'steps': [
                            {'range': [0, 30], 'color': "#FF4500"},
                            {'range': [30, 70], 'color': "#FFD700"},
                            {'range': [70, 100], 'color': "#32CD32"}
                        ],
                        'threshold': {
                            'line': {'color': "#FFFFFF", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_sentiment.update_layout(
                    height=250,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#FFFFFF"}
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
                st.markdown(f"**Sentiment:** {sentiment_text} ({sentiment_score:.2%})", unsafe_allow_html=True)
            
            with col2:
                st.markdown("##### Volatility Sentiment")
                fig_volatility = go.Figure(go.Bar(
                    x=["Volatility Sentiment"],
                    y=[volatility_score],
                    text=[f"{volatility_score:.1f}"],
                    textposition="auto",
                    marker_color="#FFD700" if volatility_score < 50 else "#FF4500",
                    hovertemplate="Volatility Score: %{y:.1f}<br>%{text}",
                    marker=dict(
                        line=dict(color="#FFFFFF", width=2)
                    )
                ))
                fig_volatility.update_layout(
                    yaxis_range=[0, 100],
                    height=250,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#FFFFFF"},
                    yaxis={'gridcolor': "rgba(255,255,255,0.2)"}
                )
                st.plotly_chart(fig_volatility, use_container_width=True)
                st.markdown(f"**Volatility Perception:** {volatility_text} ({volatility_score:.1f}/100)", unsafe_allow_html=True)
        else:
            st.warning("No news data available to analyze sentiment. Try fetching news.")
        
        st.markdown("---")
        st.markdown("*Developed by Ozy | © 2025*")

    # Tab 4: Institutional Holders

    with tab4:
        st.subheader("Institutional Holders")
        ticker = st.text_input("Ticker for Holders (e.g., AAPL):", "AAPL", key="holders_ticker").upper()
        if ticker:
            with st.spinner(f"Fetching institutional holders for {ticker}..."):
                # Usamos una solicitud directa para evitar caché y asegurar datos frescos
                url = f"{FMP_BASE_URL}/institutional-holder/{ticker}?apikey={FMP_API_KEY}"
                try:
                    response = session_fmp.get(url, headers=HEADERS_FMP, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    if not data or not isinstance(data, list):
                        st.error(f"No institutional holders data returned for {ticker}. Check ticker.")
                        logger.error(f"No data from FMP for {ticker}: {data}")
                    else:
                        holders = pd.DataFrame(data)
                        if holders.empty:
                            st.warning(f"No institutional holders data available for {ticker}.")
                        else:
                            # Verificar la fecha más reciente
                            if 'date' in holders.columns:
                                latest_date = pd.to_datetime(holders['date']).max().date()
                                st.write(f"**Latest Data Date:** {latest_date}")
                                if latest_date < datetime(2025, 3, 1).date():
                                    st.warning(f"Data is outdated (latest: {latest_date}). Expected updates beyond Dec 2024.")
                            else:
                                st.warning("")

                            # Estilizar la tabla
                            def color_negative(row):
                                if 'change' in row and row['change'] < 0:
                                    return ['color: #FF4500'] * len(row)
                                elif 'shares' in row and row['shares'] < 0:
                                    return ['color: #FF4500'] * len(row)
                                return [''] * len(row)

                            styled_holders = holders.style.apply(color_negative, axis=1).format({
                                'shares': '{:,.0f}',
                                'change': '{:,.0f}' if 'change' in holders.columns else None,
                                'value': '${:,.0f}' if 'value' in holders.columns else None,
                                'date': lambda x: x if pd.isna(x) else pd.to_datetime(x).strftime('%Y-%m-%d')
                            })
                            st.dataframe(styled_holders, use_container_width=True)

                            # Botón de descarga
                            holders_csv = holders.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Holders Data",
                                data=holders_csv,
                                file_name=f"{ticker}_institutional_holders.csv",
                                mime="text/csv",
                                key="download_tab4"
                            )
                except requests.RequestException as e:
                    st.error(f"Error fetching data for {ticker}: {str(e)}")
                    logger.error(f"HTTP error for {ticker}: {str(e)}")
            st.markdown("---")
            st.markdown("*Developed by Ozy | © 2025*")

    # Tab 5: Options Order Flow
    with tab5:
        st.subheader("Options Order Flow & Strategy Analyzer")
    
    # Input section
        col1, col2 = st.columns([1, 3])
        with col1:
            stock = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="SPY", key="stock_analysis").upper()
            expiration_dates = get_expiration_dates(stock)
            if not expiration_dates:
                st.error(f"No expiration dates found for '{stock}'. Please enter a valid stock ticker (e.g., SPY, AAPL).")
            else:
                selected_expiration = st.selectbox("Select an Expiration Date:", expiration_dates, key="stock_exp_date")
                # Toggle for Strategy Analyzer
                show_strategy_analyzer = st.checkbox("Show Options Strategy Analyzer", value=False, key="show_strategy_tab5")

    # Initialize/reset session state for strategy analyzer
    if st.session_state.get("ticker") != stock:
        st.session_state["suggested_params"] = None
        st.session_state["strategy"] = "Collar"
        st.session_state["ticker"] = stock

    with col2:
        if stock and expiration_dates:
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
                    else:
                        # Existing Metrics Section
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

                        # Existing Options Order Flow Section
                        st.markdown('<div class="section-header">Options Order Flow</div>', unsafe_allow_html=True)
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
                                title=f"| {stock} | Expiration: {selected_expiration}",
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

                        # New Options Strategy Analyzer Section
                        if show_strategy_analyzer:
                            st.markdown('<div class="section-header">Options Strategy Analyzer</div>', unsafe_allow_html=True)
                            
                            # Strategy selection
                            strategies = ["Collar", "Iron Condor", "Butterfly", "Call Spread", "Put Spread", "Buy Call", "Buy Put"]
                            strategy = st.selectbox("Select Strategy", strategies, index=strategies.index(st.session_state["strategy"]), key="strategy_select_tab5")
                            
                            # Fetch option contracts and suggest strategy
                            expiry_date = datetime.strptime(selected_expiration, "%Y-%m-%d").date()
                            option_data_strategy = get_option_contracts(stock, expiry_date)
                            max_pain_strategy = option_data_strategy["max_pain"]
                            contracts = option_data_strategy["contracts"]
                            
                            if strategy == st.session_state["strategy"]:
                                suggested_strategy, suggested_strikes = suggest_strategy(max_pain_strategy, contracts, max_pain_strategy if max_pain_strategy > 0 else current_price)
                                st.session_state["strategy"] = suggested_strategy
                                st.session_state["suggested_params"] = suggested_strikes
                            else:
                                _, suggested_strikes = suggest_strategy(max_pain_strategy, contracts, max_pain_strategy if max_pain_strategy > 0 else current_price)
                                st.session_state["suggested_params"] = suggested_strikes
                                st.session_state["strategy"] = strategy

                            # Display suggested contracts
                            st.markdown("**Suggested Contracts**")
                            if contracts:
                                for contract in contracts:
                                    st.markdown(f"- {contract['type'].capitalize()} Strike: ${contract['strike']:.2f} | Gamma: {contract['gamma']:.4f} | Volume: {contract['volume']}")
                            else:
                                st.warning(f"No contracts available for {stock} on this expiration. Using fallback strikes.")

                            # Display selected strategy
                            st.markdown(f"**Selected Strategy**: {strategy}")
                            if strategy in ["Collar", "Iron Condor", "Butterfly", "Call Spread", "Put Spread"]:
                                st.markdown(f"- Call Strike (Sell): ${suggested_strikes['call_strike']:.2f}")
                                st.markdown(f"- Put Strike (Buy): ${suggested_strikes['put_strike']:.2f}")
                                st.markdown(f"- Lower Put Strike (Sell): ${suggested_strikes['put_strike_lower']:.2f}")
                                if suggested_strikes['call_strike_higher'] > 0:
                                    st.markdown(f"- Higher Call Strike (Buy): ${suggested_strikes['call_strike_higher']:.2f}")
                            elif strategy == "Buy Call":
                                st.markdown(f"- Call Strike (Buy): ${suggested_strikes['call_strike_buy']:.2f}")
                            elif strategy == "Buy Put":
                                st.markdown(f"- Put Strike (Buy): ${suggested_strikes['put_strike_buy']:.2f}")
                            st.markdown(f"- Initial Price: ${suggested_strikes['initial_price']:.2f}")
                            st.markdown(f"- Max Pain Strike: ${max_pain_strategy:.2f}")

                            # Calculate payoff and metrics with error handling
                            try:
                                current_payoff, net_premium, current_delta, decayed_delta, current_theta = calculate_strategy_metrics(
                                    strategy, current_price, suggested_strikes["initial_price"], 
                                    suggested_strikes["call_strike"], suggested_strikes["put_strike"], 
                                    suggested_strikes["put_strike_lower"], suggested_strikes["call_strike_higher"], 
                                    expiry_date, suggested_strikes.get("call_strike_buy", 0.0), suggested_strikes.get("put_strike_buy", 0.0),
                                    stock
                                )
                            except Exception as e:
                                st.error(f"Error calculating strategy metrics: {e}")
                                logger.error(f"Failed to calculate strategy metrics: {e}")
                                current_payoff = net_premium = current_delta = decayed_delta = current_theta = 0.0

                            payoff_dollars = current_payoff * 100
                            color = "#39FF14" if current_payoff >= 0 else "#FF4500"
                            premium_color = "#39FF14" if net_premium >= 0 else "#FF4500"
                            status = "Winning" if current_payoff >= 0 else "Losing"
                            
                            st.markdown(f"<div class='metric-label'>Current Gain/Loss</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='metric-label'>Price: ${current_price:.2f} | Payoff: <span style='color:{color}'>{current_payoff:.2f} points ({status} ${payoff_dollars:,.2f})</span></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='metric-label'>Net Premium: <span style='color:{premium_color}'>${net_premium:,.2f}</span> ({'Received' if net_premium >= 0 else 'Paid'})</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='metric-label'>Delta (Current): {current_delta:.2f} | Delta (Decayed): {decayed_delta:.2f} | Theta (Daily): ${current_theta:.2f}</div>", unsafe_allow_html=True)
                            
                            # Plot payoff profile
                            st.markdown(f"<div class='metric-label'>Payoff Profile with Delta and Theta</div>", unsafe_allow_html=True)
                            try:
                                fig_strategy = plot_strategy_payoff(
                                    strategy, current_price, current_payoff, suggested_strikes["initial_price"],
                                    suggested_strikes["call_strike"], suggested_strikes["put_strike"],
                                    suggested_strikes["put_strike_lower"], suggested_strikes["call_strike_higher"],
                                    expiry_date, suggested_strikes.get("call_strike_buy", 0.0), suggested_strikes.get("put_strike_buy", 0.0),
                                    stock
                                )
                                st.plotly_chart(fig_strategy, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error plotting payoff profile: {e}")
                                logger.error(f"Failed to plot payoff profile: {e}")

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
                        volume = int(opt.get("volume", 0) or 0)
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
                        # Reemplazar NaN con 0 y manejar división por cero
                        total_buy = call_df['Total Buy ($)'].fillna(0)
                        max_total_buy = total_buy.max() if total_buy.max() > 0 else 1  # Evitar división por cero
                        sizes = np.nan_to_num(total_buy / max_total_buy * 50, nan=0, posinf=0, neginf=0)
                        sizes = np.maximum(sizes, 5)  # Asegurar un tamaño mínimo para visibilidad
                        fig.add_trace(go.Scatter(
                            x=call_df['Strike'], 
                            y=call_df['Avg Buy ($)'], 
                            mode='markers', 
                            name='CALL Buy ($)', 
                            marker=dict(
                                size=sizes, 
                                color='#228B22', 
                                opacity=0.7
                            ),
                            text=call_df['Contract'] + '<br>Total: $' + call_df['Total Buy ($)'].astype(str)
                        ))
                    put_df = df[df['Type'] == 'PUT']
                    if not put_df.empty:
                        # Reemplazar NaN con 0 y manejar división por cero
                        total_sell = put_df['Total Sell ($)'].fillna(0)
                        max_total_sell = total_sell.max() if total_sell.max() > 0 else 1  # Evitar división por cero
                        sizes = np.nan_to_num(total_sell / max_total_sell * 50, nan=0, posinf=0, neginf=0)
                        sizes = np.maximum(sizes, 5)  # Asegurar un tamaño mínimo para visibilidad
                        fig.add_trace(go.Scatter(
                            x=put_df['Strike'], 
                            y=-put_df['Avg Sell ($)'], 
                            mode='markers', 
                            name='PUT Sell ($)', 
                            marker=dict(
                                size=sizes, 
                                color='#CD5C5C', 
                                opacity=0.7
                            ),
                            text=put_df['Contract'] + '<br>Total: $' + put_df['Total Sell ($)'].astype(str)
                        ))
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
                    # Importar time explícitamente para evitar el error
                    import time
                    
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


        # Tab 9: Earnings Calendar (Tarjetas HTML con sentimiento mejorado)
    
    
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
                @st.cache_data(ttl=300)
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

                @st.cache_data(ttl=300)
                def get_vix() -> float:
                    url = f"{FMP_BASE_URL}/quote/^VIX"
                    params = {"apikey": FMP_API_KEY}
                    data = fetch_api_data(url, params, HEADERS_FMP, "VIX")
                    return float(data[0]["price"]) if data and isinstance(data, list) and "price" in data[0] else 20.0

                @st.cache_data(ttl=3600)
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
                # Usar fetch_earnings_data en lugar de get_earnings_calendar
                earnings_data = fetch_earnings_data(datetime.now().strftime("%Y-%m-%d"), (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"))
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
        # Tab 12: Performance Map
    with tab12:
        st.markdown("""
            <style>
            /* Fondo oscuro estilo terminal para toda la app */
            .stApp {
                background-color: #0A0A0A;
            }
            /* Título principal con vibe de código */
            .main-title { 
                font-size: 28px; 
                font-weight: 700; 
                color: #FFD700; /* Amarillo mostaza */
                text-align: center; 
                margin-bottom: 20px; 
                text-shadow: 0 0 10px rgba(255, 215, 0, 0.8), 0 0 20px rgba(0, 255, 0, 0.5); 
                font-family: 'Courier New', Courier, monospace;
                letter-spacing: 2px;
            }
            /* Subtítulo con verde neón */
            .section-header { 
                font-size: 20px; 
                font-weight: 600; 
                color: #39FF14; /* Verde neón */
                margin-top: 20px; 
                border-bottom: 1px dashed #00FFFF; /* Borde azul eléctrico */
                padding-bottom: 5px; 
                text-shadow: 0 0 5px rgba(57, 255, 20, 0.8); 
                font-family: 'Courier New', Courier, monospace;
            }
            /* Estilos para el contenedor de la tabla */
            div[data-testid="stTable"] {
                width: 100% !important;
                max-width: 100% !important;
                overflow-x: auto !important;
            }
            /* Estilos para la tabla con mayor especificidad */
            div[data-testid="stTable"] table.tab12-table {
                width: 100% !important;
                max-width: 100% !important;
                table-layout: fixed !important; /* Forzar el ancho de las columnas */
                border-collapse: collapse !important;
            }
            div[data-testid="stTable"] table.tab12-table th {
                background-color: #1A1F2B !important;
                color: #00FFFF !important; /* Azul eléctrico */
                font-weight: 700 !important;
                padding: 6px !important; /* Reducido para compactar */
                border: 2px solid #39FF14 !important; /* Verde neón */
                text-transform: uppercase !important;
                font-size: 10px !important; /* Reducido para compactar */
                font-family: 'Courier New', Courier, monospace !important;
                text-shadow: 0 0 3px rgba(0, 255, 255, 0.5) !important;
                line-height: 1 !important; /* Reducir espaciado vertical */
                overflow-wrap: break-word !important; /* Permitir división de texto */
                word-wrap: break-word !important;
            }
            div[data-testid="stTable"] table.tab12-table td {
                background-color: #0F1419 !important;
                color: #E0E0E0 !important; /* Blanco grisáceo */
                padding: 4px !important; /* Reducido para compactar */
                border: 2px solid #39FF14 !important; /* Verde neón */
                text-align: center !important;
                font-family: 'Courier New', Courier, monospace !important;
                font-size: 10px !important; /* Reducido para compactar */
                line-height: 1 !important; /* Reducir espaciado vertical */
                overflow-wrap: break-word !important; /* Permitir división de texto */
                word-wrap: break-word !important;
            }
            /* Botón de descarga con estilo hacker */
            .stDownloadButton button {
                background: linear-gradient(90deg, #FFD700, #39FF14); /* Amarillo mostaza a verde neón */
                color: #0A0A0A !important;
                border: 2px solid #00FFFF; /* Azul eléctrico */
                border-radius: 5px;
                padding: 8px 16px;
                font-family: 'Courier New', Courier, monospace !important;
                font-weight: 600;
                text-transform: uppercase;
                transition: all 0.3s ease;
            }
            .stDownloadButton button:hover {
                background: linear-gradient(90deg, #39FF14, #FFD700); /* Invertido al hover */
                box-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
            }
            /* Pie de página con texto gris apagado */
            .footer-text {
                color: #778DA9;
                font-size: 12px;
                text-align: center;
                font-family: 'Courier New', Courier, monospace;
                text-shadow: 0 0 2px rgba(255, 215, 0, 0.3);
            }
            </style>
        """, unsafe_allow_html=True)

        # Índices, sectores y bonos
        assets = {
            "SPY": "SPY", "Nasdaq 100": "QQQ", "Dow Jones": "DIA", "Russell 2000": "IWM",
            "Basic Materials": "XLB", "Consumer Cyclical": "XLY", "Financials": "XLF",
            "Real Estate": "XLRE", "Utilities": "XLU", "Communication": "XLC",
            "Healthcare": "XLV", "Energy": "XLE", "Industrials": "XLI", "Technology": "XLK",
            "Consumer Defensive": "XLP", "20+ Yr Treasury": "TLT", "7-10 Yr Treasury": "IEF",
            "1-3 Yr Treasury": "SHY", "VIX": "^VIX", "Dollar Index": "UUP", "FTSE 100": "EZU",
            "DAX": "DAX", "CAC 40": "EWQ", "Shanghai Comp": "ASHR", "Hang Seng": "EWH"
        }

        # Obtener datos y calcular métricas
        performance_data = []
        periods = ["1D", "1W", "1M", "1Q", "1Y"]  # Abreviados para las claves
        period_days = {"1D": 1, "1W": 5, "1M": 21, "1Q": 63, "1Y": 252}

        with st.spinner("Fetching performance data..."):
            # Obtener datos del VIX para correlación
            vix_prices, _ = get_historical_prices_combined("^VIX", limit=21)
            if len(vix_prices) < 21 or not vix_prices:
                vix_prices = [20.0] * 21  # Fallback
                st.warning("Using fallback VIX data (20.0) due to insufficient historical data.")

            for name, ticker in assets.items():
                row = {"Asset": name}

                # Precio actual desde Tradier
                current_price = get_current_price(ticker)
                if not isinstance(current_price, (int, float)) or current_price <= 0:
                    st.warning(f"Invalid current price for {ticker}: {current_price}. Using fallback price 100.0.")
                    current_price = 100.0
                row["Price"] = current_price  # Abreviado de Current_Price a Price

                # Precios históricos desde FMP
                url = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
                params = {"apikey": FMP_API_KEY, "timeseries": 260}
                try:
                    response = session_fmp.get(url, params=params, headers=HEADERS_FMP, timeout=5)
                    response.raise_for_status()
                    data = response.json()
                    if not data or "historical" not in data:
                        st.warning(f"No historical data for {ticker}. Using fallback data.")
                        prices = [current_price] * 260
                        volumes = [1000000] * 260
                    else:
                        historical = sorted(data["historical"], key=lambda x: x["date"])
                        prices = [float(day["close"]) for day in historical]
                        volumes = [int(day["volume"]) for day in historical]
                        if len(prices) < 260:
                            st.warning(f"Only {len(prices)} days of data for {ticker}. Padding with current price.")
                            prices = ([current_price] * (260 - len(prices))) + prices
                            volumes = ([1000000] * (260 - len(volumes))) + volumes
                except Exception as e:
                    st.warning(f"Error fetching historical data for {ticker}: {str(e)}. Using fallback data.")
                    prices = [current_price] * 260
                    volumes = [1000000] * 260

                # Calcular rendimientos
                for period_name, days in period_days.items():
                    if len(prices) > days:
                        initial_price = prices[-(days + 1)]
                        if initial_price <= 0:
                            row[f"{period_name}_Ret"] = np.nan
                        else:
                            row[f"{period_name}_Ret"] = calculate_performance(initial_price, current_price)
                    else:
                        row[f"{period_name}_Ret"] = np.nan

                # Métricas institucionales
                returns = np.diff(prices) / prices[:-1]
                vol_historical = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.1
                volume_avg = np.mean(volumes[-21:]) if len(volumes) >= 21 else 1
                volume_current = volumes[-1] if len(volumes) > 0 else 1
                volume_relative = volume_current / volume_avg if volume_avg > 0 else 1.0
                iv = get_implied_volatility(ticker) or vol_historical

                # Metric_IS (Institutional Score)
                momentum = row.get("1M_Ret", 0) / vol_historical if vol_historical > 0 else 0
                flow_factor = volume_relative * (iv / vol_historical if vol_historical > 0 else 1)
                year_return = row.get("1Y_Ret", 0)
                is_score = min(100, max(0, (momentum * 30 + flow_factor * 40 + (year_return / vol_historical if vol_historical > 0 else 0) * 30)))
                row["Inst_Score"] = is_score  # Abreviado de Metric_IS a Inst_Score

                # Sentiment_Label
                day_return = row.get("1D_Ret", 0)
                week_trend = row.get("1W_Ret", 0)
                sentiment_score = (day_return * 0.4 + week_trend * 0.6) * volume_relative
                row["Sentiment"] = (
                    "Bullish" if sentiment_score > 1.0 else
                    "Bearish" if sentiment_score < -1.0 else
                    "Neutral"
                )  # Abreviado de Sentiment_Label a Sentiment

                # Metric_Day_Move (%)
                vix_val = get_vix()
                vix = vix_val / 100 if vix_val is not None else 0.2
                move_factor = iv * (1 + volume_relative * 0.2) * (1 + vix * 0.5)
                day_move = move_factor * current_price * 0.15
                row["Day_Move%"] = round(day_move / current_price * 100, 2) if current_price > 0 else 0.0  # Abreviado de Metric_Day_Move (%) a Day_Move%

                # Metric_VIX_Correlation
                if len(prices) >= 21 and len(vix_prices) >= 21:
                    asset_returns_21d = np.diff(prices[-21:]) / prices[-21:-1]
                    vix_returns_21d = np.diff(vix_prices[-21:]) / vix_prices[-21:-1]
                    min_len = min(len(asset_returns_21d), len(vix_returns_21d))
                    vix_corr = np.corrcoef(asset_returns_21d[:min_len], vix_returns_21d[:min_len])[0, 1] if min_len > 0 else 0.0
                else:
                    vix_corr = 0.0
                row["VIX_Corr"] = round(vix_corr, 2)  # Abreviado de Metric_VIX_Correlation a VIX_Corr

                # Metric_Risk_Adjusted_Return
                month_return = row.get("1M_Ret", 0)
                risk_adjusted = month_return / vol_historical if vol_historical > 0 else 0.0
                row["Risk_Adj_Ret"] = round(risk_adjusted, 2)  # Abreviado de Metric_Risk_Adjusted_Return a Risk_Adj_Ret

                # Metric_Option_Volume_Spike
                try:
                    url_expirations = f"{TRADIER_BASE_URL}/markets/options/expirations"
                    params_expirations = {"symbol": ticker}
                    response_exp = session_tradier.get(url_expirations, params=params_expirations, headers=HEADERS_TRADIER, timeout=5)
                    expirations = response_exp.json().get("expirations", {}).get("date", [])
                    if not expirations:
                        raise ValueError("No expirations available")
                    nearest_expiration = sorted(expirations)[0]

                    url_options = f"{TRADIER_BASE_URL}/markets/options/chains"
                    params_options = {"symbol": ticker, "expiration": nearest_expiration}
                    response_options = session_tradier.get(url_options, params=params_options, headers=HEADERS_TRADIER, timeout=5)
                    options_data = response_options.json()
                    option_list = options_data.get("options", {}).get("option", [])
                    current_option_volume = sum(opt.get("volume", 0) for opt in option_list)
                    avg_option_volume = volume_avg * 0.1  # Proxy
                    option_spike = current_option_volume / avg_option_volume if avg_option_volume > 0 else volume_relative
                except Exception as e:
                    option_spike = volume_relative
                row["Opt_Vol_Spike"] = round(option_spike, 1)  # Abreviado de Metric_Option_Volume_Spike a Opt_Vol_Spike

                performance_data.append(row)

            if not performance_data:
                st.error("No valid data retrieved for table after processing all assets.")
                st.stop()

            # Crear DataFrame
            df = pd.DataFrame(performance_data)

            # Tabla interactiva
            st.markdown('<div class="main-title">> PERFORMANCE_MAP_</div>', unsafe_allow_html=True)

            def color_performance(val):
                if pd.isna(val):
                    return "background-color: #0F1419; color: #E0E0E0"  # Fondo oscuro, texto blanco grisáceo
                if val > 1.0:
                    intensity = min(1.0, abs(val) / 10)
                    return f"background-color: rgba(57, 255, 20, {intensity}); color: #FFFFFF"  # Verde neón
                elif val < -1.0:
                    intensity = min(1.0, abs(val) / 10)
                    return f"background-color: rgba(255, 69, 0, {intensity}); color: #FFFFFF"  # Rojo neón
                else:
                    return "background-color: rgba(255, 215, 0, 0.6); color: #0A0A0A"  # Amarillo mostaza

            def color_is(val):
                if pd.isna(val):
                    return "background-color: #0F1419; color: #E0E0E0"
                if val > 80:
                    return "background-color: rgba(57, 255, 20, 0.8); color: #FFFFFF"  # Verde neón
                elif val > 60:
                    return "background-color: rgba(255, 215, 0, 0.6); color: #0A0A0A"  # Amarillo mostaza
                else:
                    return "background-color: rgba(255, 69, 0, 0.7); color: #FFFFFF"  # Rojo neón

            def color_sentiment(val):
                if val == "Bullish":
                    return "background-color: rgba(57, 255, 20, 0.8); color: #FFFFFF"  # Verde neón
                elif val == "Bearish":
                    return "background-color: rgba(255, 69, 0, 0.7); color: #FFFFFF"  # Rojo neón
                else:
                    return "background-color: rgba(0, 255, 255, 0.6); color: #0A0A0A"  # Azul eléctrico

            def color_vix_corr(val):
                if pd.isna(val):
                    return "background-color: #0F1419; color: #E0E0E0"
                if val < -0.5:
                    return "background-color: rgba(57, 255, 20, 0.8); color: #FFFFFF"  # Verde neón
                elif val > 0.5:
                    return "background-color: rgba(255, 69, 0, 0.7); color: #FFFFFF"  # Rojo neón
                else:
                    return "background-color: rgba(255, 215, 0, 0.6); color: #0A0A0A"  # Amarillo mostaza

            def color_risk_adj(val):
                if pd.isna(val):
                    return "background-color: #0F1419; color: #E0E0E0"
                if val > 0.2:
                    return "background-color: rgba(57, 255, 20, 0.8); color: #FFFFFF"  # Verde neón
                elif val < -0.2:
                    return "background-color: rgba(255, 69, 0, 0.7); color: #FFFFFF"  # Rojo neón
                else:
                    return "background-color: rgba(0, 255, 255, 0.6); color: #0A0A0A"  # Azul eléctrico

            def color_option_spike(val):
                if pd.isna(val):
                    return "background-color: #0F1419; color: #E0E0E0"
                if val > 2.0:
                    return "background-color: rgba(57, 255, 20, 0.8); color: #FFFFFF"  # Verde neón
                elif val < 1.0:
                    return "background-color: rgba(255, 69, 0, 0.7); color: #FFFFFF"  # Rojo neón
                else:
                    return "background-color: rgba(255, 215, 0, 0.6); color: #0A0A0A"  # Amarillo mostaza

            styled_df = df.style.format({
                "1D_Ret": "{:.2f}%", "1W_Ret": "{:.2f}%", "1M_Ret": "{:.2f}%",
                "1Q_Ret": "{:.2f}%", "1Y_Ret": "{:.2f}%", "Inst_Score": "{:.1f}",
                "Day_Move%": "{:.2f}%", "VIX_Corr": "{:.2f}",
                "Risk_Adj_Ret": "{:.2f}", "Opt_Vol_Spike": "{:.1f}",
                "Price": "{:.2f}"
            }, na_rep="N/A").applymap(color_performance, subset=[f"{p}_Ret" for p in periods] + ["Day_Move%"]).applymap(color_is, subset=["Inst_Score"]).applymap(color_sentiment, subset=["Sentiment"]).applymap(color_vix_corr, subset=["VIX_Corr"]).applymap(color_risk_adj, subset=["Risk_Adj_Ret"]).applymap(color_option_spike, subset=["Opt_Vol_Spike"]).set_properties(**{
                "text-align": "center", "border": "2px solid #39FF14", "font-family": "'Courier New', Courier, monospace", "font-size": "10px", "padding": "4px"
            }).set_table_styles([
                {"selector": "th", "props": [("background-color", "#1A1F2B"), ("color", "#00FFFF"), ("font-weight", "700"), ("text-align", "center"), ("border", "2px solid #39FF14"), ("padding", "6px"), ("font-family", "'Courier New', Courier, monospace")]}
            ])

            st.dataframe(styled_df, use_container_width=True, height=1000)

            csv = df.to_csv(index=False)
            st.download_button(
                label="> DOWNLOAD_DATA_",
                data=csv,
                file_name="performance_table_map.csv",
                mime="text/csv",
                key="download_tab12"
            )

            st.markdown(f'<div class="footer-text">> LAST_UPDATED: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | POWERED_BY_OZY_ANALYTICS_</div>', unsafe_allow_html=True)

    
if __name__ == "__main__":
    main()
