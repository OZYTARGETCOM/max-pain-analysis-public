import streamlit as st  # Para la interfaz Streamlit
import pandas as pd  # Para manipulación de datos tabulares
import requests  # Para llamadas a la API Tradier
import plotly.express as px  # Para gráficos interactivos sencillos
import plotly.graph_objects as go  # Para gráficos avanzados
from datetime import datetime, timedelta  # Para manejo de fechas
import numpy as np  # Para cálculos matemáticos y manipulación de arrays
import csv
import bcrypt
import os
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import socket



# Archivo para contraseñas
PASSWORDS_FILE = "passwords.csv"

# Inicializar archivo de contraseñas con 20 contraseñas predefinidas
def initialize_passwords_file():
    if not os.path.exists(PASSWORDS_FILE):
        with open(PASSWORDS_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            passwords = [
                ["abc123", 0, ""],
                ["def456", 0, ""],
                ["ghi789", 0, ""],
                ["jkl012", 0, ""],
                ["mno345", 0, ""],
                ["pqr678", 0, ""],
                ["stu901", 0, ""],
                ["vwx234", 0, ""],
                ["yz1234", 0, ""],
                ["abcd56", 0, ""],
                ["efgh78", 0, ""],
                ["ijkl90", 0, ""],
                ["mnop12", 0, ""],
                ["qrst34", 0, ""],
                ["uvwx56", 0, ""],
                ["yzab78", 0, ""],
                ["cdef90", 0, ""],
                ["ghij12", 0, ""],
                ["klmn34", 0, ""],
                ["opqr56", 0, ""],
            ]
            writer.writerows(passwords)

# Obtener la dirección IP local
def get_local_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e:
        return None

# Cargar contraseñas
def load_passwords():
    passwords = {}
    try:
        with open(PASSWORDS_FILE, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 3:  # Validar formato correcto
                    password, status, ip = row
                    passwords[password] = {"status": int(status), "ip": ip}
    except Exception:
        pass
    return passwords

# Guardar contraseñas
def save_passwords(passwords):
    with open(PASSWORDS_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        for password, data in passwords.items():
            writer.writerow([password, data["status"], data["ip"]])

# Autenticar contraseña
def authenticate_password(input_password):
    local_ip = get_local_ip()
    if not local_ip:
        st.error("No se pudo obtener la IP local.")
        return False

    passwords = load_passwords()

    if input_password in passwords:
        password_data = passwords[input_password]
        if password_data["status"] == 0:
            # Primera vez que se usa la contraseña
            passwords[input_password]["status"] = 1
            passwords[input_password]["ip"] = local_ip
            save_passwords(passwords)
            return True
        elif password_data["status"] == 1 and password_data["ip"] == local_ip:
            # Contraseña ya usada, pero desde la misma IP
            return True
        elif password_data["status"] == 1 and password_data["ip"] != local_ip:
            st.warning("⚠️ Esta contraseña ya ha sido usada desde otra dirección IP.")
            return False
    return False  # Contraseña incorrecta

# Inicializar archivo de contraseñas
initialize_passwords_file()

# Manejo de sesión
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Pantalla de autenticación
if not st.session_state["authenticated"]:
    st.title("🔒 Acceso Restringido")
    password = st.text_input("Ingresa tu contraseña", type="password")
    if st.button("Iniciar Sesión"):
        if authenticate_password(password):
            st.session_state["authenticated"] = True
    else:
        st.error("❌ Contraseña incorrecta.")
    st.stop()  # Detener la ejecución si no está autenticado

# Contenido principal de la aplicación (solo si está autenticado)









################################################app

# Tradier API Configuration
API_KEY = "wMG8GrrZMBFeZMCWJTqTzZns7B4w"
BASE_URL = "https://api.tradier.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}

# Function: Get expiration dates
def get_expiration_dates(ticker):
    url = f"{BASE_URL}/markets/options/expirations"
    params = {"symbol": ticker}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get("expirations", {}).get("date", [])
    else:
        st.error("Error fetching expiration dates.")
        return []

# Function: Get current price
def get_current_price(ticker):
    url = f"{BASE_URL}/markets/quotes"
    params = {"symbols": ticker}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get("quotes", {}).get("quote", {}).get("last", 0)
    else:
        st.error("Error fetching current price.")
        return 0

# Function: Analyze relevant contracts
def analyze_contracts(ticker, expiration, current_price):
    url = f"{BASE_URL}/markets/options/chains"
    params = {
        "symbol": ticker,
        "expiration": expiration,
        "greeks": True
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error("Error fetching contracts.")
        return pd.DataFrame()

    options = response.json().get("options", {}).get("option", [])
    if not options:
        st.warning("No contracts found.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(options)

    # Handle missing important columns
    for col in ['strike', 'option_type', 'last', 'iv', 'volume', 'delta', 'gamma', 'theta', 'vega']:
        if col not in df.columns:
            df[col] = 0

    # Filter top 10 contracts by IV and volume
    df['strike_diff'] = abs(df['strike'] - current_price)
    df = df.sort_values(by=["iv", "volume"], ascending=[False, False]).head(10)

    return df

# Function: Calculate movement potential
def calculate_potential(df, current_price):
    """
    Adds columns for:
    - Break-even point
    - Percentage of required movement
    """
    if df.empty:
        return df

    # Calculate break-even point
    df['break_even'] = df.apply(
        lambda row: row['strike'] + row['last'] if row['option_type'] == 'call' else row['strike'] - row['last'],
        axis=1
    )

    # Calculate required movement to reach break-even
    df['% movement'] = ((df['break_even'] - current_price) / current_price * 100).round(2)

    return df

# Function: Style the table to highlight Calls and Puts
def style_table(df):
    def highlight_row(row):
        color = 'background-color: green; color: white;' if row['option_type'] == 'call' else 'background-color: red; color: white;'
        return [color] * len(row)

    return df.style.apply(highlight_row, axis=1).format({
        'strike': '{:.2f}',
        'last': '${:.2f}',
        'iv': '{:.2%}',
        'volume': '{:,}',
        'delta': '{:.2f}',
        'gamma': '{:.6f}',
        'theta': '{:.2f}',
        'vega': '{:.2f}',
        'break_even': '${:.2f}',
        '% movement': '{:.2f}%'
    })

# Function: Select suggested contracts
def select_best_contracts(df, current_price):
    if df.empty:
        return None, None

    # Closest contract to the current price
    df['strike_diff'] = abs(df['strike'] - current_price)
    closest_contract = df.sort_values(
        by=['strike_diff', 'iv', 'volume'],
        ascending=[True, False, False]
    ).iloc[0]

    # Economical contract with potential
    otm_df = df[(df['strike'] > current_price) & (df['last'] < 5)]
    economic_contract = (
        otm_df.sort_values(by=['iv', 'volume'], ascending=[False, False]).iloc[0]
        if not otm_df.empty else None
    )

    return closest_contract, economic_contract

# Function: Summarize contracts (Calls vs Puts)
def summarize_contracts(df):
    if df.empty:
        return {
            'Total Calls': 0,
            'Total Puts': 0,
            'Total Call Premium': 0.0,
            'Total Put Premium': 0.0
        }

    calls = df[df['option_type'] == 'call']
    puts = df[df['option_type'] == 'put']

    def format_large_number(value):
        if value >= 1e9:
            return f"{value / 1e9:.2f}B"
        elif value >= 1e6:
            return f"{value / 1e6:.2f}M"
        else:
            return f"{value:.6f}"

    summary = {
        'Total Calls': len(calls),
        'Total Puts': len(puts),
        'Total Call Premium': format_large_number(calls['last'].sum() if not calls['last'].isnull().all() else 0.0),
        'Total Put Premium': format_large_number(puts['last'].sum() if not puts['last'].isnull().all() else 0.0)
    }

    return summary


# Visual separator
st.divider()

# Step 1: Enter Ticker
st.header("1️⃣ Enter Ticker")
ticker = st.text_input("🔎 Enter the underlying ticker:", value="SPY")

if ticker:
    # Step 2: Automatically Fetch Expiration Dates
    st.header("2️⃣ Select Expiration Date")
    st.write("Fetching expiration dates...")
    expirations = get_expiration_dates(ticker)

    if expirations:
        # Display expiration dates in a dropdown
        selected_expiration = st.selectbox("📅 Select an expiration date:", expirations)

        # Step 3: Get Current Price of Underlying
        st.header("3️⃣ Current Price")
        current_price = get_current_price(ticker)
        st.markdown(f"**💰 Current Price:** **${current_price:.2f}**")

        # Step 4: Analyze Contracts
        st.header("4️⃣ Analysis Results")
        if st.button("📊 Analyze Contracts"):
            st.write("Processing contract analysis...")
            df = analyze_contracts(ticker, selected_expiration, current_price)

            if not df.empty:
                # Add potential calculations to the DataFrame
                df = calculate_potential(df, current_price)

                # Contract Summary
                summary = summarize_contracts(df)
                st.subheader("📊 Contract Summary")
                st.markdown(f"""
                - **Total Calls:** {summary['Total Calls']}  
                - **Total Puts:** {summary['Total Puts']}  
                - **Total Call Premium:** {summary['Total Call Premium']}  
                - **Total Put Premium:** {summary['Total Put Premium']}
                """)

                # Display styled table with additional calculations
                st.subheader("🔝 Top Options Contracts")
                st.dataframe(style_table(df))

                # Suggested contracts
                closest_contract, economic_contract = select_best_contracts(df, current_price)

                if closest_contract is not None:
                    st.subheader("✅ POTENTIAL  OPTIONS")
                    st.markdown(f"""
                        **Strike:** {closest_contract['strike']}  
                        **Type:** {closest_contract['option_type']}  
                        **Contract Price:** ${closest_contract['last']:.2f}  
                        **Break-Even:** ${closest_contract['break_even']:.2f}  
                        **% Required Movement:** {closest_contract['% movement']:.2f}%
                    """)

                if economic_contract is not None:
                    st.subheader("💡 Economical Contract with Potential")
                    st.markdown(f"""
                        **Strike:** {economic_contract['strike']}  
                        **Type:** {economic_contract['option_type']}  
                        **Contract Price:** ${economic_contract['last']:.2f}  
                        **Break-Even:** ${economic_contract['break_even']:.2f}  
                        **% Required Movement:** {economic_contract['% movement']:.2f}%
                    """)
                else:
                    st.warning("No economical contracts found.")
            else:
                st.warning("No relevant contracts found.")





#############################################################SEGURIDAD  ARRIVA     



































# Configuración de la API Tradier
API_KEY = "wMG8GrrZMBFeZMCWJTqTzZns7B4w"
BASE_URL = "https://api.tradier.com/v1"

# Función para obtener datos de opciones
@st.cache_data(ttl=30)
def get_options_data(ticker, expiration_date):
    
    url = f"{BASE_URL}/markets/options/chains"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker, "expiration": expiration_date, "greeks": "true"}
    response = requests.get(url, headers=headers, params=params)
    

    if response.status_code == 200:
        return response.json().get("options", {}).get("option", [])
    else:
        st.error("Error fetching options data.")
        return []
    
import time
time.sleep(0.2)  # Pausa de 200ms entre solicitudes




@st.cache_data(ttl=30)
def get_historical_data(ticker, days=10):
    url = f"{BASE_URL}/markets/history"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker, "interval": "daily", "start": pd.Timestamp.now().date() - pd.Timedelta(days=days)}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        history = response.json().get("history", {}).get("day", [])
        return [day["close"] for day in history]
    else:
        st.error("Error fetching historical data.")
        return []

@st.cache_data(ttl=30)
def get_expiration_dates(ticker):
    url = f"{BASE_URL}/markets/options/expirations"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("expirations", {}).get("date", [])
    else:
        st.error("Error fetching expiration dates.")
        return []

@st.cache_data(ttl=30)
def get_current_price(ticker):
    url = f"{BASE_URL}/markets/quotes"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbols": ticker}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        quote = response.json().get("quotes", {}).get("quote", {})
        return quote.get("last", 0)
    else:
        st.error("Error fetching current price.")
        return 0

# Detectar strikes tocados (sube y baja en 10 días)
def detect_touched_strikes(strikes, historical_prices):
    touched_strikes = set()
    for strike in strikes:
        for i in range(1, len(historical_prices)):
            if (historical_prices[i-1] < strike <= historical_prices[i]) or (historical_prices[i-1] > strike >= historical_prices[i]):
                touched_strikes.add(strike)
    return touched_strikes



# Función para obtener fechas de expiración
@st.cache_data(ttl=30)
def get_expiration_dates(ticker):
    url = f"{BASE_URL}/markets/options/expirations"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json().get("expirations", {}).get("date", [])
    else:
        st.error("Error fetching expiration dates.")
        return []

# Función para obtener el precio actual
@st.cache_data(ttl=30)
def get_current_price(ticker):
    url = f"{BASE_URL}/markets/quotes"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbols": ticker}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        quote = response.json().get("quotes", {}).get("quote", {})
        return quote.get("last", 0)
    else:
        st.error("Error fetching current price.")
        return 0

# Función para calcular Max Pain ajustado
# Función optimizada para calcular el Max Pain
def calculate_max_pain_optimized(options_data):
    if not options_data:
        return None

    # Diccionario para agrupar datos por strike
    strikes = {}
    for option in options_data:
        strike = option["strike"]
        oi = option.get("open_interest", 0) or 0
        volume = option.get("volume", 0) or 0
        option_type = option["option_type"].upper()

        if strike not in strikes:
            strikes[strike] = {"CALL": {"OI": 0, "Volume": 0}, "PUT": {"OI": 0, "Volume": 0}}

        # Acumular OI y Volumen
        strikes[strike][option_type]["OI"] += oi
        strikes[strike][option_type]["Volume"] += volume

    # Lista de strikes ordenados
    strike_prices = sorted(strikes.keys())

    # Calcular la pérdida total para cada strike
    total_losses = {}
    for strike in strike_prices:
        loss_call = sum(
            (strikes[s]["CALL"]["OI"] + strikes[s]["CALL"]["Volume"]) * max(0, s - strike)
            for s in strike_prices
        )
        loss_put = sum(
            (strikes[s]["PUT"]["OI"] + strikes[s]["PUT"]["Volume"]) * max(0, strike - s)
            for s in strike_prices
        )
        total_losses[strike] = loss_call + loss_put

    # Strike con la menor pérdida total
    max_pain = min(total_losses, key=total_losses.get)
    return max_pain

# Modificar el gráfico de Gamma Exposure para usar el cálculo mejorado
# Función para crear el gráfico de exposición gamma optimizado
# Función para obtener datos de opciones
@st.cache_data(ttl=30)
def get_options_data(ticker, expiration_date):
    url = f"{BASE_URL}/markets/options/chains"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker, "expiration": expiration_date, "greeks": "true"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("options", {}).get("option", [])
    else:
        st.error("Error fetching options data.")
        return []

# Función para obtener fechas de expiración
@st.cache_data(ttl=30)
def get_expiration_dates(ticker):
    url = f"{BASE_URL}/markets/options/expirations"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("expirations", {}).get("date", [])
    else:
        st.error("Error fetching expiration dates.")
        return []

# Función para obtener el precio actual
@st.cache_data(ttl=30)
def get_current_price(ticker):
    url = f"{BASE_URL}/markets/quotes"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbols": ticker}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        quote = response.json().get("quotes", {}).get("quote", {})
        return quote.get("last", 0)
    else:
        st.error("Error fetching current price.")
        return 0

# Función optimizada para calcular el Max Pain
def calculate_max_pain_optimized(options_data):
    if not options_data:
        return None

    strikes = {}
    for option in options_data:
        strike = option["strike"]
        oi = option.get("open_interest", 0) or 0
        volume = option.get("volume", 0) or 0
        option_type = option["option_type"].upper()

        if strike not in strikes:
            strikes[strike] = {"CALL": {"OI": 0, "Volume": 0}, "PUT": {"OI": 0, "Volume": 0}}

        # Acumular OI y Volumen
        strikes[strike][option_type]["OI"] += oi
        strikes[strike][option_type]["Volume"] += volume

    strike_prices = sorted(strikes.keys())
    total_losses = {}
    for strike in strike_prices:
        loss_call = sum((strikes[s]["CALL"]["OI"] + strikes[s]["CALL"]["Volume"]) * max(0, s - strike) for s in strike_prices)
        loss_put = sum((strikes[s]["PUT"]["OI"] + strikes[s]["PUT"]["Volume"]) * max(0, strike - s) for s in strike_prices)
        total_losses[strike] = loss_call + loss_put

    return min(total_losses, key=total_losses.get)

# Gráfico con Max Pain y Expiración
# Gráfico dinámico de Gamma Exposure
def gamma_exposure_chart(processed_data, current_price, touched_strikes):
    strikes = sorted(processed_data.keys())
    
    # Calcular Gamma CALLs y PUTs
    gamma_calls = [
        processed_data[s]["CALL"]["OI"] * processed_data[s]["CALL"]["Gamma"] * current_price 
        for s in strikes
    ]
    gamma_puts = [
        -processed_data[s]["PUT"]["OI"] * processed_data[s]["PUT"]["Gamma"] * current_price 
        for s in strikes
    ]
    
    # Verificar los strikes tocados y asignar colores dinámicos
    call_colors = ["grey" if s in touched_strikes else "#7DF9FF" for s in strikes]  # Gamma CALL
    put_colors = ["orange" if s in touched_strikes else "red" for s in strikes]    # Gamma PUT

    fig = go.Figure()

    # Añadir Gamma CALLs
    fig.add_trace(go.Bar(
        x=strikes,
        y=gamma_calls,
        name="Gamma CALL",
        marker=dict(color=call_colors),
        hovertemplate="<b>Strike:</b> %{x}<br><b>Gamma CALL:</b> %{y:.2f}<extra></extra>"
    ))

    # Añadir Gamma PUTs
    fig.add_trace(go.Bar(
        x=strikes,
        y=gamma_puts,
        name="Gamma PUT",
        marker=dict(color=put_colors),
        hovertemplate="<b>Strike:</b> %{x}<br><b>Gamma PUT:</b> %{y:.2f}<extra></extra>"
    ))

    # Línea para el precio actual
    fig.add_shape(
        type="line",
        x0=current_price, x1=current_price,
        y0=min(gamma_calls + gamma_puts) * 1.1,
        y1=max(gamma_calls + gamma_puts) * 1.1,
        line=dict(color="#39FF14", dash="dot", width=1),  # Línea punteada
    )

    # Etiqueta del precio actual
    fig.add_annotation(
        x=current_price,
        y=max(gamma_calls + gamma_puts) * 1.05,
        text=f"Current Price: {current_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#39FF14",
        font=dict(color="#39FF14", size=12)
    )

    # Configuración de hover label visible
    fig.update_traces(hoverlabel=dict(
        bgcolor="rgba(30,30,30,0.9)",
        bordercolor="white",
        font=dict(color="white", size=12)
    ))

    fig.update_layout(
        title="|SCANNER|",
        xaxis_title="VOL",
        yaxis_title="VOLUME",
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig



# Función para crear Heatmap
# Opción para colores personalizados
default_colorscale = [
    [0, "#5A0000"],  # Rojo oscuro
    [0.25, "#3B528B"],  # Azul oscuro
    [0.5, "#21918C"],  # Verde
    [0.75, "#5DC863"],  # Verde claro
    [1, "#FDE725"]  # Amarillo
]

# Función para crear el Heatmap sin Theta y Delta
def create_heatmap(processed_data, current_price, max_pain, custom_colorscale=None):
    strikes = sorted(processed_data.keys())

    # Calculamos métricas clave
    volume = [
        processed_data[s]["CALL"]["OI"] * processed_data[s]["CALL"]["Gamma"] +
        processed_data[s]["PUT"]["OI"] * processed_data[s]["PUT"]["Gamma"] for s in strikes
    ]
    gamma = [processed_data[s]["CALL"]["Gamma"] + processed_data[s]["PUT"]["Gamma"] for s in strikes]
    oi = [processed_data[s]["CALL"]["OI"] + processed_data[s]["PUT"]["OI"] for s in strikes]

    data = pd.DataFrame({
        'Volume': volume,
        'Gamma': gamma,
        'OI': oi
    })

    data_normalized = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Usar Viridis por defecto o la escala personalizada
    colorscale = custom_colorscale if custom_colorscale else "Viridis"

    fig = go.Figure(data=go.Heatmap(
        z=data_normalized.T.values,
        x=strikes,
        y=data_normalized.columns,
        colorscale=colorscale,
        colorbar=dict(title='Normalized Value'),
        hoverongaps=False
    ))

    fig.update_layout(
        title={
            "text": f"GEAR|Price: ${current_price:.2f} |MM TARGET${max_pain}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis_title="Strikes",
        yaxis_title="GEAR",
        template="plotly_dark"
    )

    # Añadir líneas de referencia
    fig.add_shape(
        type="line",
        x0=current_price, x1=current_price,
        y0=0, y1=len(data_normalized.columns) - 1,
        line=dict(color="orange", dash="dot"),
        name="Current Price"
    )

    fig.add_shape(
        type="line",
        x0=max_pain, x1=max_pain,
        y0=0, y1=len(data_normalized.columns) - 1,
        line=dict(color="green", dash="dash"),
        name="Max Pain"
    )

    # Agregar marcadores para las áreas clave
    targets = {
        "Gamma": {"values": gamma, "color": "red", "symbol": "γ"},
        "Volume": {"values": volume, "color": "blue", "symbol": "🔧"},
        "OI": {"values": oi, "color": "orange", "symbol": "OI"}
    }

    for metric, details in targets.items():
        metric_values = details["values"]
        color = details["color"]
        symbol = details["symbol"]

        # Seleccionar el valor más relevante según lógica:
        top_index = max(range(len(metric_values)), key=lambda i: metric_values[i])

        strike = strikes[top_index]
        fig.add_annotation(
            x=strike,
            y=metric,  # Posición dinámica en el eje Y
            text=symbol,  # Símbolo del marcador
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=color,
            ax=0,  # Offset horizontal
            ay=-50,  # Offset vertical
            font=dict(color=color, size=12),
        )

    return fig







# Función para crear Skew Analysis Chart
def plot_skew_analysis_with_totals(options_data):
    # Crear listas para strikes, IV y tipos
    strikes = [option["strike"] for option in options_data]
    iv = [option.get("implied_volatility", 0) * 100 for option in options_data]
    option_type = [option["option_type"].upper() for option in options_data]
    open_interest = [option.get("open_interest", 0) for option in options_data]

    # Sumar el Open Interest y Volumen total para CALLS y PUTS
    total_calls = sum(option.get("open_interest", 0) for option in options_data if option["option_type"].upper() == "CALL")
    total_puts = sum(option.get("open_interest", 0) for option in options_data if option["option_type"].upper() == "PUT")
    total_volume_calls = sum(option.get("volume", 0) for option in options_data if option["option_type"].upper() == "CALL")
    total_volume_puts = sum(option.get("volume", 0) for option in options_data if option["option_type"].upper() == "PUT")

    # Aplicar desplazamiento dinámico en el eje Y
    adjusted_iv = [
        iv[i] + (open_interest[i] * 0.01) if option_type[i] == "CALL" else
        -(iv[i] + (open_interest[i] * 0.01)) for i in range(len(iv))
    ]

    # Crear DataFrame para análisis
    skew_df = pd.DataFrame({
        "Strike": strikes,
        "Adjusted IV (%)": adjusted_iv,
        "Option Type": option_type,
        "Open Interest": open_interest
    })

    # Crear gráfico interactivo con Plotly Express
    title = f"IV Analysis<br><span style='font-size:16px;'> CALLS: {total_calls} | PUTS: {total_puts} | VC {total_volume_calls} | VP {total_volume_puts}</span>"
    fig = px.scatter(
        skew_df,
        x="Strike",
        y="Adjusted IV (%)",
        color="Option Type",
        size="Open Interest",
        hover_data=["Strike", "Option Type", "Open Interest", "Adjusted IV (%)"],
        title=title,
        labels={"Option Type": "Contract Type"},
        color_discrete_map={"CALL": "blue", "PUT": "red"},
    )

    # Ajustar diseño del gráfico
    fig.update_layout(
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (%) (CALLS y PUTS)",
        legend_title="Option Type",
        template="plotly_white",
        title_x=0.5  # Centrar el título
    )
    return fig, total_calls, total_puts






# Interfaz de usuario
st.title("SCANNER")

ticker = st.text_input("Ticker", value="SPY", key="ticker_input").upper()
expiration_dates = get_expiration_dates(ticker)
if expiration_dates:
    expiration_date = st.selectbox("Expiration Date", expiration_dates, key="expiration_date")
else:
    st.error("No expiration dates available.")
    st.stop()

current_price = get_current_price(ticker)
options_data = get_options_data(ticker, expiration_date)
if not options_data:
    st.error("No options data available.")
    st.stop()

import time
time.sleep(0.2)  # Pausa de 200ms entre solicitudes


# Calcular Max Pain con el cálculo mejorado
# Calcular Max Pain con el cálculo mejorado
max_pain = calculate_max_pain_optimized(options_data)

# Procesar datos para gráficos con validaciones
processed_data = {}

for opt in options_data:
    # Verificar si el elemento es válido
    if not opt or not isinstance(opt, dict):
        continue  # Ignorar valores inválidos

    # Validar y obtener valores seguros
    strike = opt.get("strike")
    if not isinstance(strike, (int, float)):
        continue  # Ignorar si el strike no es válido

    option_type = opt.get("option_type", "").upper()
    if option_type not in ["CALL", "PUT"]:
        continue  # Ignorar si el tipo de opción no es válido

    greeks = opt.get("greeks", {})
    gamma = greeks.get("gamma", 0) if isinstance(greeks, dict) else 0
    open_interest = opt.get("open_interest", 0)

    # Inicializar estructura si no existe
    if strike not in processed_data:
        processed_data[strike] = {"CALL": {"Gamma": 0, "OI": 0}, "PUT": {"Gamma": 0, "OI": 0}}

    # Actualizar datos
    processed_data[strike][option_type]["Gamma"] += gamma
    processed_data[strike][option_type]["OI"] += open_interest

# Validar si hay datos procesados
if not processed_data:
    st.error("No valid data to display Gamma Exposure.")
    st.stop()

# Mostrar gráficos



# Interfaz de Usuario




current_price = get_current_price(ticker)
historical_prices = get_historical_data(ticker, days=10)

options_data = get_options_data(ticker, expiration_date)
processed_data = {}

if options_data:
    for opt in options_data:
        strike = opt["strike"]
        option_type = opt["option_type"].upper()
        oi = opt.get("open_interest", 0)
        gamma = opt.get("greeks", {}).get("gamma", 0)

        if strike not in processed_data:
            processed_data[strike] = {"CALL": {"OI": 0, "Gamma": 0}, "PUT": {"OI": 0, "Gamma": 0}}

        processed_data[strike][option_type]["OI"] += oi
        processed_data[strike][option_type]["Gamma"] += gamma

    # Detectar strikes tocados por cruce
    touched_strikes = detect_touched_strikes(processed_data.keys(), historical_prices)

    # Generar gráfico
    gamma_fig = gamma_exposure_chart(processed_data, current_price, touched_strikes)
    st.plotly_chart(gamma_fig, use_container_width=True)
else:
    st.error("No options data available.")

############################################################




# Interfaz de usuario






# Procesar datos para gráficos
processed_data = {}
for opt in options_data:
    strike = opt["strike"]
    option_type = opt["option_type"].upper()
    gamma = opt.get("greeks", {}).get("gamma", 0)
    delta = opt.get("greeks", {}).get("delta", 0)
    theta = opt.get("greeks", {}).get("theta", 0)
    open_interest = opt.get("open_interest", 0)

    if strike not in processed_data:
        processed_data[strike] = {"CALL": {"Gamma": 0, "OI": 0, "Delta": 0, "Theta": 0}, "PUT": {"Gamma": 0, "OI": 0, "Delta": 0, "Theta": 0}}

    processed_data[strike][option_type]["Gamma"] = gamma
    processed_data[strike][option_type]["OI"] = open_interest
    processed_data[strike][option_type]["Delta"] = delta
    processed_data[strike][option_type]["Theta"] = theta

# Calcular Max Pain
def calculate_max_pain(options_data):
    strikes = {}
    for option in options_data:
        strike = option["strike"]
        oi = option.get("open_interest", 0) or 0
        option_type = option["option_type"].upper()

        if strike not in strikes:
            strikes[strike] = {"CALL": 0, "PUT": 0}

        strikes[strike][option_type] += oi

    total_losses = {}
    strike_prices = sorted(strikes.keys())

    for strike in strike_prices:
        loss_call = sum((strikes[s]["CALL"] * max(0, s - strike)) for s in strike_prices)
        loss_put = sum((strikes[s]["PUT"] * max(0, strike - s)) for s in strike_prices)
        total_losses[strike] = loss_call + loss_put

    return min(total_losses, key=total_losses.get)

max_pain = calculate_max_pain(options_data)

# Crear y mostrar el Heatmap
heatmap_fig = create_heatmap(processed_data, current_price, max_pain)
st.plotly_chart(heatmap_fig, use_container_width=True)






st.subheader("Options")

# Llamar a la función mejorada
skew_fig, total_calls, total_puts = plot_skew_analysis_with_totals(options_data)

# Mostrar los totales en Streamlit
st.write(f"**Total CALLS** {total_calls}")
st.write(f"**Total PUTS** {total_puts}")

# Mostrar el gráfico
st.plotly_chart(skew_fig, use_container_width=True)























# Función para generar señales en formato de tarjetas





#########################################################################








#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>







################################################################################































































# Interfaz de usuario



current_price = get_current_price(ticker)
options_data = get_options_data(ticker, expiration_date)
import time
time.sleep(0.2)  # Pausa de 200ms entre solicitudes


if not options_data:
    st.error("No options data available.")
    st.stop()

# Calcular niveles clave y alertarrent_price, key_levels)

# Mostrar resultados
st.subheader("Current Price")
st.markdown(f"**${current_price:.2f}**")



# Visualización de Gamma y OI
def plot_gamma_oi(key_levels):
    strikes = []
    gammas = []
    open_interests = []
    option_types = []

    for option_type, levels in key_levels.items():
        for strike, gamma, _, oi in levels:
            strikes.append(strike)
            gammas.append(gamma)
            open_interests.append(oi)
            option_types.append(option_type)

    df = pd.DataFrame({"Strike": strikes, "Gamma": gammas, "OI": open_interests, "Type": option_types})
    fig = px.scatter(df, x="Strike", y="Gamma", size="OI", color="Type", title="Executed Strike")
    return fig











############################################################################################










all_tickers = [
    # 100 Tickers de NASDAQ
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "ADBE", "INTC", "NFLX",
    "QCOM", "CSCO", "AMD", "PYPL", "AVGO", "AMAT", "TXN", "MRVL", "INTU", "SHOP",
    "JD", "ZM", "DOCU", "CRWD", "SNOW", "ZS", "PANW", "SPLK", "MDB", "OKTA",
    "ROKU", "ALGN", "ADSK", "DXCM", "TEAM", "PDD", "MELI", "BIDU", "BABA", "NTES",
    "ATVI", "EA", "ILMN", "EXPE", "SIRI", "KLAC", "LRCX", "ASML", "SWKS", "XLNX",
    "WDAY", "TTWO", "VRTX", "REGN", "BIIB", "SGEN", "MAR", "CTSH", "FISV", "MTCH",
    "TTD", "SPLK", "PTON", "DOCS", "UPST", "HIMS", "CRSP", "NVCR", "EXAS", "ARKK",
    "ZS", "TWLO", "U", "HUBS", "VIX", "BILL", "ZI", "GTLB", "NET", "FVRR",
    "TTD", "COIN", "RBLX", "DKNG", "SPOT", "SNAP", "PINS", "MTCH", "LYFT", "GRPN",

    # 100 Tickers de NYSE
    "BRK.B", "JNJ", "V", "PG", "JPM", "HD", "DIS", "MA", "UNH", "PFE", "KO", "PEP",
    "BAC", "WMT", "XOM", "CVX", "ABT", "TMO", "MRK", "MCD", "CAT", "GS", "MMM",
    "RTX", "IBM", "DOW", "GE", "BA", "LMT", "FDX", "T", "VZ", "NKE", "AXP", "ORCL",
    "CSX", "USB", "SPG", "AMT", "PLD", "CCI", "PSA", "CB", "BK", "SCHW", "TFC", "SO",
    "D", "DUK", "NEE", "EXC", "SRE", "AEP", "EIX", "PPL", "PEG", "FE", "AEE", "AES",
    "ETR", "XEL", "AWK", "WEC", "ED", "ES", "CNP", "CMS", "DTE", "EQT", "OGE",
    "OKE", "SWX", "WMB", "APA", "DVN", "FANG", "MRO", "PXD", "HAL", "SLB", "COP",
    "CVX", "XOM", "PSX", "MPC", "VLO", "HES", "OXY", "EOG", "KMI", "WES","DJT","BITX","SMCI","ENPH",

    # 100 Tickers de Russell
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

# Función para obtener datos de múltiples tickers
@st.cache_data(ttl=30)
def fetch_batch_stock_data(tickers):
    tickers_str = ",".join(tickers)
    url = f"{BASE_URL}/markets/quotes"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbols": tickers_str}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json().get("quotes", {}).get("quote", [])
        if isinstance(data, dict):
            data = [data]
        return [
            {
                "Ticker": item.get("symbol", ""),
                "Price": item.get("last", 0),
                "Change (%)": item.get("change_percentage", 0),
                "Volume": item.get("volume", 0),
                "Average Volume": item.get("average_volume", 1),
                "IV": item.get("implied_volatility", None),
                "HV": item.get("historical_volatility", None),
                "Previous Close": item.get("prev_close", 0)
            }
            for item in data
        ]
    else:
        st.error(f"Error al obtener datos: {response.status_code}")
        return []



# Función para calcular potencial explosivo
def calculate_explosive_movers(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    # Rellenar valores nulos y ajustar tipos
    df["IV"] = pd.to_numeric(df["IV"], errors='coerce').fillna(0)
    df["HV"] = pd.to_numeric(df["HV"], errors='coerce').fillna(0)
    df["Average Volume"] = pd.to_numeric(df["Average Volume"], errors='coerce').replace(0, np.nan)

    # Calcular métricas
    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Explosión"] = df["Volumen Relativo"] * df["Change (%)"].abs()
    df["Score"] = df["Explosión"] + (df["IV"] * 0.5)

    return df.sort_values("Score", ascending=False).head(3)

# Función para calcular actividad de opciones
def calculate_options_activity(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    # Rellenar valores nulos y ajustar tipos
    df["IV"] = pd.to_numeric(df["IV"], errors='coerce').fillna(0)
    df["Average Volume"] = pd.to_numeric(df["Average Volume"], errors='coerce').replace(0, np.nan)

    # Calcular actividad de opciones
    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Options Activity"] = df["Volumen Relativo"] * df["IV"]

    return df.sort_values("Options Activity", ascending=False).head(3)


##################################################################################################################


import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

# Función para buscar noticias en Google
def fetch_google_news(keywords):
    base_url = "https://www.google.com/search"
    query = "+".join(keywords)
    params = {"q": query, "tbm": "nws", "tbs": "qdr:h"}  # Última hora

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

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

# Función para buscar noticias en Bing
def fetch_bing_news(keywords):
    base_url = "https://www.bing.com/news/search"
    query = " ".join(keywords)
    params = {"q": query, "qft": "+filterui:age-lt24h"}  # Últimas 24 horas

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

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

# Función para buscar publicaciones en Instagram
def fetch_instagram_posts(keywords):
    base_url = "https://www.instagram.com/explore/tags/"
    posts = []

    for keyword in keywords:
        if keyword.startswith("#"):
            try:
                url = f"{base_url}{keyword[1:]}/"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
                }

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

# Configuración de Streamlit
st.title(" News Scanner ")


keywords = st.text_input("Enter keywords (comma-separated Boludo!!!!):", "Trump, ElonMusk").split(",")
keywords = [k.strip() for k in keywords if k.strip()]

if "news_data" not in st.session_state:
    st.session_state.news_data = []

news_placeholder = st.empty()

with st.spinner("Fetching breaking news..."):
    google_news = fetch_google_news(keywords)
    bing_news = fetch_bing_news(keywords)
    instagram_posts = fetch_instagram_posts(keywords)

latest_news = google_news + bing_news + instagram_posts

if latest_news:
    st.session_state.news_data = latest_news
    with news_placeholder.container():
        st.success(f"Latest news updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!")
        for idx, article in enumerate(latest_news, 1):
            st.markdown(f"### {idx}. [{article['title']}]({article['link']})")
            st.markdown(f"**Published:** {article['time']}\n")
            st.markdown("---")
else:
    st.error("No recent news found from any source.")




import streamlit as st
import webbrowser

def generate_ticker_search_url(ticker):
    base_url = "https://x.com/search?q="
    query = f"%24{ticker}"  # Agregar el prefijo '$' para tickers
    return f"{base_url}{query}&f=live"

# Configuración de Streamlit


# Entrada de ticker
ticker = st.text_input(
    "Buscar en X ",
    "",
    placeholder="Escribe el ticker y presiona Enter..."
).strip().upper()

# Abrir automáticamente si se ingresa un ticker válido
if ticker:
    search_url = generate_ticker_search_url(ticker)
    webbrowser.open_new_tab(search_url)  # Abrir el enlace en el navegador
    st.stop()  # Detener la ejecución para evitar recargar
