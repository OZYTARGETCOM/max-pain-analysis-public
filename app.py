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
import plotly.express as px
import plotly.graph_objects as go



# Archivo para contraseñas
PASSWORDS_FILE = "passwords.csv"

# Inicializar archivo de contraseñas con 20 contraseñas predefinidas
def initialize_passwords_file():
    if not os.path.exists(PASSWORDS_FILE):
        with open(PASSWORDS_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            passwords = [
                ["abc123", 0, ""],#ozy
                ["def456", 0, ""],#hector
                ["ghi789", 0, ""],#silvana
                ["jkl010", 0, ""],#pereira
                ["mno345", 0, ""],#joege
                ["pqr678", 0, ""],#sandra
                ["stu901", 0, ""],#minu
                ["vwx234", 0, ""],#
                ["yz1234", 0, ""],#
                ["abcd56", 0, ""],#gusman
                ["efgh78", 0, ""],
                ["ijkl90", 0, ""],
                ["mnop12", 0, ""],#MARCK
                ["qrst34", 0, ""],
                ["uvwx56", 0, ""],
                ["yzab78", 0, ""],
                ["cdef90", 0, ""],#mary
                ["ghij12", 0, ""],
                ["news34", 0, ""],#
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
    st.title("🔒 Acceso VIP")
    password = st.text_input("Ingresa tu contraseña", type="password")
    if st.button("Iniciar Sesión"):
        if authenticate_password(password):
            st.session_state["authenticated"] = True
    else:
        st.error("❌ VIP access only with Monitor/Indicator clients.")
    st.stop()  # Detener la ejecución si no está autenticado

# Contenido principal de la aplicación (solo si está autenticado)








################################################app
################################################app


# Tradier API Configuration
API_KEY = "d0H5QGsma6Bh41VBw6P6lItCBl7D"
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
        st.error("Error retrieving expiration dates.")
        return []

# Function: Get current underlying price
def get_current_price(ticker):
    url = f"{BASE_URL}/markets/quotes"
    params = {"symbols": ticker}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get("quotes", {}).get("quote", {}).get("last", 0)
    else:
        st.error("Error retrieving the current price.")
        return 0

# Function: Analyze relevant option contracts
def analyze_contracts(ticker, expiration, current_price):
    url = f"{BASE_URL}/markets/options/chains"
    params = {
        "symbol": ticker,
        "expiration": expiration,
        "greeks": True
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        st.error("Error retrieving option contracts.")
        return pd.DataFrame()

    options = response.json().get("options", {}).get("option", [])
    if not options:
        st.warning("No contracts available.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(options)

    # Handle missing critical columns
    for col in ['strike', 'option_type', 'open_interest', 'volume', 'bid', 'ask', 'last_volume',
                'trade_date', 'bid_exchange', 'delta', 'gamma', 'break_even']:
        if col not in df.columns:
            df[col] = 0  # Assign default if missing

    # Add placeholder data for missing fields
    df['trade_date'] = datetime.now().strftime('%Y-%m-%d')  # Placeholder trade date
    df['break_even'] = df.apply(
        lambda row: row['strike'] + row['bid'] if row['option_type'] == 'call' else row['strike'] - row['bid'],
        axis=1
    )

    return df

# Function: Style and sort table with the requested column order
def style_and_sort_table(df):
    ordered_columns = ['strike', 'option_type', 'open_interest', 'volume', 'trade_date', 
                       'bid', 'ask', 'last_volume', 'bid_exchange', 'delta', 'gamma', 'break_even']
    df = df.sort_values(by=['volume', 'open_interest'], ascending=[False, False]).head(10)
    df = df[ordered_columns]

    def highlight_row(row):
        color = 'background-color: green; color: white;' if row['option_type'] == 'call' else 'background-color: red; color: white;'
        return [color] * len(row)

    return df.style.apply(highlight_row, axis=1).format({
        'strike': '{:.2f}',
        'bid': '${:.2f}',
        'ask': '${:.2f}',
        'last_volume': '{:,}',
        'open_interest': '{:,}',
        'delta': '{:.2f}',
        'gamma': '{:.2f}',
        'break_even': '${:.2f}'
    })

# Function: Select recommended contracts
def select_best_contracts(df, current_price):
    if df.empty:
        return None, None
    df['strike_diff'] = abs(df['strike'] - current_price)
    closest_contract = df.sort_values(
        by=['strike_diff', 'volume', 'open_interest'],
        ascending=[True, False, False]
    ).iloc[0]

    otm_calls = df[(df['option_type'] == 'call') & (df['strike'] > current_price) & (df['ask'] < 5)]
    otm_puts = df[(df['option_type'] == 'put') & (df['strike'] < current_price) & (df['ask'] < 5)]

    if not otm_calls.empty or not otm_puts.empty:
        economic_df = pd.concat([otm_calls, otm_puts])
        economic_contract = economic_df.sort_values(
            by=['volume', 'open_interest'], ascending=[False, False]
        ).iloc[0]
    else:
        economic_contract = None

    return closest_contract, economic_contract

# Function: Calculate Max Pain
def calculate_max_pain(df):
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
    max_pain_strike = max_pain_df.loc[max_pain_df['total_loss'].idxmin()]
    return max_pain_strike, max_pain_df.sort_values(by='total_loss', ascending=True)

# Function: Calculate Support, Resistance, and Mid Level
def calculate_support_resistance_mid(max_pain_table, current_price):
    # Filtrar puts y calls
    puts = max_pain_table[max_pain_table['strike'] <= current_price]
    calls = max_pain_table[max_pain_table['strike'] > current_price]

    # Identificar soporte basado en interés abierto de puts
    if not puts.empty:
        support_level = puts.loc[puts['total_loss'].idxmin()]['strike']
    else:
        support_level = current_price  # Por defecto al precio actual si no hay datos

    # Identificar resistencia basada en interés abierto de calls
    if not calls.empty:
        resistance_level = calls.loc[calls['total_loss'].idxmin()]['strike']
    else:
        resistance_level = current_price  # Por defecto al precio actual si no hay datos

    # Calcular nivel medio
    mid_level = (support_level + resistance_level) / 2

    return support_level, resistance_level, mid_level


# Function: Plot Histogram with Support, Resistance, and Mid Level
def plot_max_pain_histogram_with_levels(max_pain_table, current_price):
    support_level, resistance_level, mid_level = calculate_support_resistance_mid(max_pain_table, current_price)

    max_pain_table['loss_category'] = max_pain_table['total_loss'].apply(
        lambda x: 'High Loss' if x > max_pain_table['total_loss'].quantile(0.75) else
                  ('Low Loss' if x < max_pain_table['total_loss'].quantile(0.25) else 'Neutral')
    )

    color_map = {
        'High Loss': '#FF5733',
        'Low Loss': '#28A745',
        'Neutral': 'rgba(128,128,128,0.3)'
    }

    fig = px.bar(
        max_pain_table,
        x='total_loss',
        y='strike',
        orientation='h',
        title="Max Pain Histogram with Levels",
        labels={'total_loss': 'Total Loss', 'strike': 'Strike Price'},
        color='loss_category',
        color_discrete_map=color_map
    )

    fig.update_layout(
        xaxis_title="Total Loss",
        yaxis_title="Strike Price",
        template="plotly_white",
        font=dict(size=14, family="Open Sans"),
        title=dict(
            text="📊 Max Pain Analysis ",
            font=dict(size=18),
            x=0.5
        ),
        hovermode="y",
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="#FFFF00",
            spikethickness=1.5
        )
    )

    mean_loss = max_pain_table['total_loss'].mean()
    fig.add_vline(
        x=mean_loss,
        line_width=1,
        line_dash="dash",
        line_color="#00FF00",
        annotation_text=f"Mean Loss: {mean_loss:.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#00FF00", size=12)
    )

    fig.add_hline(
        y=support_level,
        line_width=1,
        line_dash="dot",
        line_color="#1E90FF",
        annotation_text=f"Support: {support_level:.2f}",
        annotation_position="bottom right",
        annotation_font=dict(color="#1E90FF", size=10)
    )

    fig.add_hline(
        y=resistance_level,
        line_width=1,
        line_dash="dot",
        line_color="#FF4500",
        annotation_text=f"Resistance: {resistance_level:.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#FF4500", size=10)
    )

    fig.add_hline(
        y=mid_level,
        line_width=1,
        line_dash="solid",
        line_color="#FFD700",
        annotation_text=f"Mid Level: {mid_level:.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#FFD700", size=8)
    )

    return fig
# Function: Get option chains
def get_option_chains(ticker, expiration):
    url = f"{BASE_URL}/markets/options/chains"
    params = {
        "symbol": ticker,
        "expiration": expiration,
        "greeks": True
    }
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json().get("options", {}).get("option", [])
    else:
        st.error("Error retrieving option chains.")
        return []

# Function: Calculate Score
def calculate_score(df, current_price, volatility=0.2):
    df['score'] = (df['open_interest'] * df['volume']) / (abs(df['strike'] - current_price) + volatility)
    return df.sort_values(by='score', ascending=False)

# Function: Display Cards
def display_cards(df):
    st.markdown("### Top 5")
    for i, row in df.iterrows():
        st.markdown(f"""
        **Strike:** {row['strike']}  
        **Type:** {'Call' if row['option_type'] == 'call' else 'Put'}  
        **Volume:** {row['volume']}  
        **Open Interest:** {row['open_interest']}  
        **Score:** {row['score']:.2f}  
        """)

# Function: Plot Histogram with Enhanced Visualization
def plot_histogram(df):
    fig = px.bar(
        df,
        x='strike',
        y='score',
        color='option_type',
        title="Score by Strike (Calls and Puts)",
        labels={'score': 'Relevance Score', 'strike': 'Strike Price'},
        text='score',
        color_discrete_map={
            'call': '#00FF00',  # Fosforescente verde para Calls
            'put': '#FF00FF'   # Fosforescente magenta para Puts
        }
    )
    fig.update_traces(
        texttemplate='%{text:.2f}', 
        textposition='outside', 
        marker=dict(line=dict(width=0.5, color='black'))  # Bordes delgados
    )
    fig.update_layout(
        plot_bgcolor='black',  # Fondo negro para resaltar colores
        font=dict(color='white', size=12),  # Fuente blanca para contraste
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='gray'),
        xaxis_title="Strike Price",
        yaxis_title="Relevance Score"
    )

    # Líneas de soporte y resistencia
    support_level = df['strike'].iloc[0]  # Ejemplo: primer strike como soporte
    resistance_level = df['strike'].iloc[-1]  # Ejemplo: último strike como resistencia

    fig.add_hline(
        y=support_level, 
        line_width=1,  # Línea más delgada
        line_dash="dot", 
        line_color="#1E90FF",
        annotation_text=f"Support: {support_level:.2f}",
        annotation_position="bottom left",
        annotation_font=dict(size=10, color="#1E90FF")
    )
    fig.add_hline(
        y=resistance_level, 
        line_width=1,  # Línea más delgada
        line_dash="dot", 
        line_color="#FF4500",
        annotation_text=f"Resistance: {resistance_level:.2f}",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#FF4500")
    )

    st.plotly_chart(fig)


# Streamlit Interface
st.divider()

# Step 1: Enter Ticker

        



#############################################################SEGURIDAD  ARRIVA     












# Configuración de la API Tradier
API_KEY = "d0H5QGsma6Bh41VBw6P6lItCBl7D"
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


# Función para crear el Heatmap sin Theta y Delta








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
st.title("News Scanner")

keywords = st.text_input("Enter keywords (comma-separated Boludo!!!!):", "Trump").split(",")
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
        
        # Agregar un expander para las noticias
        with st.expander("📰 Latest News"):
            st.write("Click to expand and view the latest news.")
            for idx, article in enumerate(latest_news, 1):
                st.markdown(f"### {idx}. [{article['title']}]({article['link']})")
                st.markdown(f"**Published:** {article['time']}\n")
                st.markdown("---")
else:
    st.error("No recent news found from any source.")




import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

# --- Suprimir Advertencias ---
import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

# --- API Configuration ---
TRADIER_API_KEY = "d0H5QGsma6Bh41VBw6P6lItCBl7D"
FMP_API_KEY = "bQ025fPNVrYcBN4KaExd1N3Xczyk44wM"

TRADIER_BASE_URL = "https://api.tradier.com/v1"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

HEADERS = {
    "Authorization": f"Bearer {TRADIER_API_KEY}",
    "Accept": "application/json"
}

# --- Cache to Avoid Repeated API Calls ---
@st.cache_data(ttl=3600)
def get_stock_list():
    """Fetch a filtered list of stocks using FMP's stock-screener endpoint"""
    try:
        response = requests.get(
            f"{FMP_BASE_URL}/stock-screener",
            params={
                "apikey": FMP_API_KEY,
                "marketCapMoreThan": 1_000_000_000,  # Market cap > $1B
                "volumeMoreThan": 500_000,          # Volume > 500,000
                "priceMoreThan": 10,               # Price > $10
                "priceLessThan": 100,              # Price < $100
                "betaMoreThan": 1                  # High volatility stocks
            }
        )
        response.raise_for_status()
        data = response.json()
        return [stock["symbol"] for stock in data]
    except Exception as e:
        st.error(f"❌ Error fetching stock list: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_historical_prices(symbol, period="daily", limit=30):
    """Fetch historical prices for a stock"""
    try:
        response = requests.get(
            f"{TRADIER_BASE_URL}/markets/history",
            headers=HEADERS,
            params={"symbol": symbol, "interval": period, "limit": limit}
        )
        response.raise_for_status()
        data = response.json()
        
        # Validate that the response contains valid historical data
        if not data or "history" not in data or "day" not in data["history"]:
            st.warning(f"⚠️ No historical data for {symbol}")
            return [], []
        
        prices = [float(day["close"]) for day in data["history"]["day"]]
        volumes = [int(day["volume"]) for day in data["history"]["day"]]
        return prices, volumes
    except Exception as e:
        st.warning(f"⚠️ Error fetching historical data for {symbol}: {str(e)}")
        return [], []

def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
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
    """Calculate Simple Moving Average (SMA)"""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def scan_stock(symbol, scan_type, breakout_period=10, volume_threshold=2.0):
    """Scan a single stock and return results based on scan type"""
    prices, volumes = get_historical_prices(symbol)
    
    if len(prices) <= breakout_period or len(volumes) == 0:
        return None  # Skip stocks with insufficient data
    
    rsi = calculate_rsi(prices)
    sma = calculate_sma(prices)
    avg_volume = np.mean(volumes)
    current_volume = volumes[-1]
    recent_high = max(prices[-breakout_period:])
    recent_low = min(prices[-breakout_period:])
    last_price = prices[-1]
    
    near_support = abs(last_price - recent_low) / recent_low <= 0.05  # ±5%
    near_resistance = abs(last_price - recent_high) / recent_high <= 0.05  # ±5%
    
    breakout_type = None
    if last_price > recent_high:
        breakout_type = "Up"
    elif last_price < recent_low:
        breakout_type = "Down"
    
    possible_change = None
    if near_support:
        possible_change = (recent_low - last_price) / last_price * 100
    elif near_resistance:
        possible_change = (recent_high - last_price) / last_price * 100
    
    if scan_type == "Bullish (Upward Momentum)" and sma is not None and last_price > sma and rsi is not None and rsi < 70:
        return {
            "Symbol": symbol,
            "Last Price": last_price,
            "SMA": round(sma, 2),
            "RSI": round(rsi, 2),
            "Near Support": near_support,
            "Near Resistance": near_resistance,
            "Volume": current_volume,
            "Breakout Type": breakout_type,
            "Possible Change (%)": round(possible_change, 2) if possible_change else None
        }
    elif scan_type == "Bearish (Downward Momentum)" and sma is not None and last_price < sma and rsi is not None and rsi > 30:
        return {
            "Symbol": symbol,
            "Last Price": last_price,
            "SMA": round(sma, 2),
            "RSI": round(rsi, 2),
            "Near Support": near_support,
            "Near Resistance": near_resistance,
            "Volume": current_volume,
            "Breakout Type": breakout_type,
            "Possible Change (%)": round(possible_change, 2) if possible_change else None
        }
    elif scan_type == "Breakouts":
        if breakout_type:
            return {
                "Symbol": symbol,
                "Breakout Type": breakout_type,
                "Last Price": last_price,
                "Recent High": recent_high,
                "Recent Low": recent_low,
                "Volume": current_volume,
                "Possible Change (%)": round(possible_change, 2) if possible_change else None
            }
        elif near_support or near_resistance:
            return {
                "Symbol": symbol,
                "Potential Breakout": "Support" if near_support else "Resistance",
                "Last Price": last_price,
                "Recent High": recent_high,
                "Recent Low": recent_low,
                "Volume": current_volume,
                "Possible Change (%)": round(possible_change, 2) if possible_change else None
            }
    elif scan_type == "Unusual Volume" and current_volume > volume_threshold * avg_volume:
        return {
            "Symbol": symbol,
            "Volume": current_volume,
            "Avg Volume": avg_volume,
            "Last Price": last_price
        }
    return None

# --- User Interface ---
#st.set_page_config(page_title="📈 Quantum Scanner Pro", layout="wide", page_icon="📊")

# Custom CSS for Professional Look
st.markdown("""
<style>
.centered {
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
#st.markdown('<div class="centered"><h1>📊 Quantum Scanner Pro</h1></div>', unsafe_allow_html=True)

# Scanning Parameters (centrado en lugar de en el sidebar)
st.markdown('<div class="centered"><h2>⚙️ Scanning Parameters</h2></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])  # Dividimos la pantalla en dos columnas para organizar mejor los elementos

with col1:
    scan_type = st.radio(
        "Select Scan Type:",
        ["Bullish (Upward Momentum)", "Bearish (Downward Momentum)", 
         "Breakouts", "Volume Unusual"],
        index=0
    )

with col2:
    max_results = st.slider("Max Stocks to Display:", 1, 200, 30)
    if scan_type == "Breakouts":
        breakout_period = st.slider("Breakout Period (Days):", 5, 30, 10)

# Main Input (Botón centrado)
st.markdown('<div class="centered">', unsafe_allow_html=True)
if st.button("🚀 Start Scan", key="scan_button", help="Click to start scanning the market"):
    st.info("ℹ️ Scanning the market...")
    # Fetch Filtered Stock List
    stock_list = get_stock_list()
    if stock_list:
        st.write(f"Filtered {len(stock_list)} stocks based on predefined criteria.")
        results = []
        progress_bar = st.progress(0)
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(
                    scan_stock,
                    symbol,
                    scan_type,
                    breakout_period if scan_type == "Breakouts" else 10,
                    volume_threshold=2.0 if scan_type == "Abnormal Volume" else None
                )
                for symbol in stock_list
            ]
            for i, future in enumerate(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / len(stock_list))
        if results:
            st.markdown('🚀 Scan Results', unsafe_allow_html=True)
            df_results = pd.DataFrame(results[:max_results])
            # Style the DataFrame
            styled_df = df_results.style \
                .background_gradient(cmap="Blues") \
                .set_properties(**{"text-align": "center"}) \
                .set_table_styles([{
                    "selector": "th",
                    "props": [("font-size", "16px"), ("text-align", "center"), ("color", "#2E86C1")]
                }])
            st.dataframe(styled_df, use_container_width=True)
            # Generate Dynamic Histogram for Volumes
            if "Volume" in df_results.columns:
                fig = go.Figure()
                for _, row in df_results.iterrows():
                    color = "green" if row.get("Breakout Type") == "Up" else "red" if row.get("Breakout Type") == "Down" else "blue"
                    hover_text = (
                        f"Symbol: {row['Symbol']}\n"
                        f"Volume: {row['Volume']:,}\n"
                        f"Breakout Type: {row.get('Breakout Type', 'None')}\n"
                        f"Possible Change: {row.get('Possible Change (%)', 'N/A')}%"
                    )
                    fig.add_trace(go.Bar(
                        x=[row["Symbol"]],
                        y=[row["Volume"]],
                        text=row["Volume"],
                        textposition="auto",
                        marker_color=color,
                        hoverinfo="text",
                        hovertext=hover_text
                    ))
                fig.update_layout(
                    title="📊 Volume Distribution of Scanned Stocks",
                    xaxis_title="Stock Symbol",
                    yaxis_title="Volume",
                    showlegend=False,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Volume data not available for histogram.")
        else:
            st.warning("⚠️ No stocks match the specified criteria.")
    else:
        st.error("❌ Unable to fetch stock list.")

st.markdown('</div>', unsafe_allow_html=True)

# Export Button
if "results" in locals() and results:
    csv = pd.DataFrame(results).to_csv(index=False)
    st.download_button(
        label="📥 Export Results to CSV",
        data=csv,
        file_name="quantum_scanner_results.csv",
        mime="text/csv"
    )






















    import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xml.etree.ElementTree as ET  # Para manejar respuestas XML
from typing import Dict, List, Optional  # Para especificar tipos de datos

# --- Suprimir Advertencias ---
import logging
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

# --- API Configuration ---
TRADIER_API_KEY = "d0H5QGsma6Bh41VBw6P6lItCBl7D"
TRADIER_BASE_URL = "https://api.tradier.com/v1"
FMP_API_KEY = "bQ025fPNVrYcBN4KaExd1N3Xczyk44wM"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# --- Función para obtener fechas de vencimiento desde Tradier (XML) ---
def get_expiration_dates(symbol):
    response = requests.get(
        f"{TRADIER_BASE_URL}/markets/options/expirations",
        params={"symbol": symbol},
        headers={"Authorization": f"Bearer {TRADIER_API_KEY}"},
    )
    if response.status_code != 200:
        st.error(f"Error al obtener las fechas de vencimiento. Código de estado: {response.status_code}")
        st.write("Respuesta de la API:", response.text)
        return []
    
    try:
        root = ET.fromstring(response.text)
        expiration_dates = [date.text for date in root.findall(".//date")]
        if not expiration_dates:
            st.error("La respuesta de la API no contiene fechas de vencimiento.")
            return []
        return expiration_dates
    except ET.ParseError:
        st.error("La respuesta de la API no es un XML válido.")
        st.write("Respuesta de la API:", response.text)
        return []

# --- Función para obtener datos de opciones desde Tradier (XML) ---
def get_option_data(symbol, expiration_date):
    response = requests.get(
        f"{TRADIER_BASE_URL}/markets/options/chains",
        params={"symbol": symbol, "expiration": expiration_date},
        headers={"Authorization": f"Bearer {TRADIER_API_KEY}"},
    )
    if response.status_code != 200:
        st.error(f"Error al obtener los datos de opciones. Código de estado: {response.status_code}")
        st.write("Respuesta de la API:", response.text)
        return pd.DataFrame()
    
    try:
        root = ET.fromstring(response.text)
        options = []
        for option in root.findall(".//option"):
            # Determinar acción basada en bid y ask
            bid = float(option.find("bid").text) if option.find("bid").text else 0
            ask = float(option.find("ask").text) if option.find("ask").text else 0
            action = "buy" if bid > 0 and ask > 0 else "sell"

            option_data = {
                "symbol": option.find("symbol").text,
                "description": option.find("description").text,
                "type": option.find("option_type").text,
                "strike": float(option.find("strike").text),
                "open_interest": int(option.find("open_interest").text),
                "action": action,
            }
            options.append(option_data)
        if not options:
            st.error("Datos de opciones.")
            return pd.DataFrame()
        return pd.DataFrame(options)
    except ET.ParseError:
        st.error("La respuesta de la API no es un XML válido.")
        st.write("Respuesta de la API:", response.text)
        return pd.DataFrame()

# --- Cache to Avoid Repeated API Calls ---
@st.cache_data(ttl=3600)
def get_financial_metrics(symbol: str) -> Dict[str, float]:
    """Fetch financial metrics like EBITDA, ROE, etc."""
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
            "Current Price": latest_quote.get("price", 0),
            "EBITDA": latest_income.get("ebitda", 0),
            "Revenue": latest_income.get("revenue", 0),
            "Net Income": latest_income.get("netIncome", 0),
            "ROA": latest_metrics.get("roa", 0),
            "ROE": latest_metrics.get("roe", 0),
            "Beta": latest_metrics.get("beta", 0),
            "PE Ratio": latest_metrics.get("peRatio", 0),
            "Debt-to-Equity Ratio": latest_metrics.get("debtToEquity", 0),
            "Dividend Yield": latest_metrics.get("dividendYield", 0),
            "Working Capital": latest_balance.get("totalCurrentAssets", 0) - latest_balance.get("totalCurrentLiabilities", 0),
            "Total Assets": latest_balance.get("totalAssets", 0),
            "Retained Earnings": latest_balance.get("retainedEarnings", 0),
            "EBIT": latest_income.get("ebit", 0),
            "Market Cap": latest_metrics.get("marketCap", 0),
            "Total Liabilities": latest_balance.get("totalLiabilities", 0),
            "Operating Cash Flow": latest_cash_flow.get("operatingCashFlow", 0),
            "Current Ratio": latest_metrics.get("currentRatio", 0),
            "Long Term Debt": latest_balance.get("longTermDebt", 0),
            "Shares Outstanding": latest_metrics.get("sharesOutstanding", 0),
            "Gross Margin": latest_metrics.get("grossProfitMargin", 0),
            "Asset Turnover": latest_metrics.get("assetTurnover", 0),
            "Capital Expenditure": latest_cash_flow.get("capitalExpenditure", 0),
            "Free Cash Flow": latest_cash_flow.get("freeCashFlow", 0),
            "Weighted Average Shares Diluted": latest_income.get("weightedAverageShsOutDil", 0),
            "Property Plant Equipment Net": latest_balance.get("propertyPlantEquipmentNet", 0),
            "Cash and Cash Equivalents": latest_balance.get("cashAndCashEquivalents", 0),
            "Total Debt": latest_balance.get("totalDebt", 0),
            "Interest Expense": latest_income.get("interestExpense", 0),
            "Dividend Paid": latest_cash_flow.get("dividendPaid", 0),
            "Short Term Debt": latest_balance.get("shortTermDebt", 0),
            "Intangible Assets": latest_balance.get("intangibleAssets", 0),
            "Accounts Receivable": latest_balance.get("accountsReceivable", 0),
            "Inventory": latest_balance.get("inventory", 0),
            "Accounts Payable": latest_balance.get("accountsPayable", 0),
            "COGS": latest_income.get("costOfRevenue", 0),
            "Tax Rate": latest_income.get("incomeTaxExpense", 0) / latest_income.get("incomeBeforeTax", 1) if latest_income.get("incomeBeforeTax", 1) != 0 else 0,
        }
    except Exception as e:
        st.warning(f"⚠️ Error fetching financial metrics for {symbol}: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def get_historical_prices(symbol: str, period: str = "daily", limit: int = 30) -> (List[float], List[int]):
    """Fetch historical prices."""
    try:
        response = requests.get(
            f"{FMP_BASE_URL}/historical-price-full/{symbol}?apikey={FMP_API_KEY}&timeseries={limit}"
        )
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

def calculate_sma(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Simple Moving Average (SMA)."""
    if len(prices) < period:
        return None
    return np.mean(prices[-period:])

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index (RSI)."""
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
    rsi = 100 - (100 / (1 + rs)) if rs != float('inf') else 100
    return rsi

def speculate_next_day_movement(metrics: Dict[str, float], prices: List[float], volumes: List[int]) -> (str, float, Optional[float]):
    """
    Speculate the next day's price movement based on financial metrics and historical data.
    """
    sma = calculate_sma(prices, period=50)
    rsi = calculate_rsi(prices, period=14)
    recent_high = max(prices[-10:]) if len(prices) >= 10 else None
    recent_low = min(prices[-10:]) if len(prices) >= 10 else None
    last_price = prices[-1] if prices else None
    avg_volume = np.mean(volumes[-10:]) if len(volumes) >= 10 else None
    current_volume = volumes[-1] if volumes else None

    # Default values if any metric is missing
    trend = "High Volatility"
    confidence = 0.5  # Neutral confidence by default

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

    # Fundamental Analysis
    if metrics.get("ROE", 0) > 0.15 and metrics.get("Free Cash Flow", 0) > 0:
        confidence += 0.1
    if metrics.get("Current Ratio", 0) < 1:
        confidence -= 0.1
    if metrics.get("Beta", 0) > 1.5:
        confidence += 0.1 if trend == "Bullish" else -0.1

    # Predict Next Day Movement
    predicted_change = (last_price * 0.01) * confidence if trend == "Bullish" else -(last_price * 0.01) * confidence
    predicted_price = last_price + predicted_change if last_price is not None else None
    return trend, confidence, predicted_price

# --- Función para aplicar estilos dinámicos a la tabla ---
def style_option_data(option_data):
    def highlight_calls_puts(row):
        # Colorear filas según el tipo de opción (call o put)
        if row["type"] == "call":
            return ["background-color: rgba(0, 255, 0, 0.2)"] * len(row)  # Verde claro para calls
        elif row["type"] == "put":
            return ["background-color: rgba(255, 0, 0, 0.2)"] * len(row)  # Rojo claro para puts
        else:
            return [""] * len(row)

    def highlight_actions(row):
        # Resaltar acciones (buy o sell)
        if row["action"] == "buy":
            return ["color: green"] * len(row)  # Texto verde para buy
        elif row["action"] == "sell":
            return ["color: orange"] * len(row)  # Texto naranja para sell
        else:
            return [""] * len(row)

    def highlight_high_open_interest(row):
        # Resaltar strikes con interés abierto alto
        threshold = option_data["open_interest"].quantile(0.75)  # Percentil 75
        styles = []
        for value in row:
            if isinstance(value, (int, float)) and value > threshold:
                styles.append("background-color: yellow")  # Amarillo para valores altos
            else:
                styles.append("")
        return styles

    # Aplicar múltiples estilos
    styled_df = (
        option_data.style.apply(highlight_calls_puts, axis=1)
        .apply(highlight_actions, axis=1)
        .apply(highlight_high_open_interest, axis=1)
    )
    return styled_df

# --- Streamlit App ---
st.title("📈 SCANNER PRO")

# Input: Stock Ticker
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", value="AAPL").upper()
selected_expiration = st.selectbox("Select an Expiration Date:", expiration_dates)

if stock:
    # Fetch Data
    financial_metrics = get_financial_metrics(stock)
    prices, volumes = get_historical_prices(stock)
    if not prices or not volumes:
        st.error(f"❌ Unable to fetch data for {stock}.")
    else:
        # Calculate additional metrics
        trend, confidence, predicted_price = speculate_next_day_movement(financial_metrics, prices, volumes)

        # Display Metrics
        st.markdown(f"### Metrics for {stock}")
        st.write(f"- **Current Price**: ${financial_metrics.get('Current Price', 0):,.2f}")
        st.write(f"- **EBITDA**: ${financial_metrics.get('EBITDA', 0):,.2f}")
        st.write(f"- **Revenue**: ${financial_metrics.get('Revenue', 0):,.2f}")
        st.write(f"- **Net Income**: ${financial_metrics.get('Net Income', 0):,.2f}")
        st.write(f"- **ROE**: {financial_metrics.get('ROE', 0):.2f}")
        st.write(f"- **Beta**: {financial_metrics.get('Beta', 0):.2f}")
        st.write(f"- **PE Ratio**: {financial_metrics.get('PE Ratio', 0):.2f}")
        st.write(f"- **Debt-to-Equity Ratio**: {financial_metrics.get('Debt-to-Equity Ratio', 0):.2f}")
        st.write(f"- **Dividend Yield**: {financial_metrics.get('Dividend Yield', 0):.2%}")
        st.write(f"- **Market Cap**: ${financial_metrics.get('Market Cap', 0):,.2f}")

        # Display Speculation
        st.markdown(f"### Speculation for {stock}")
        st.write(f"- **Current Price**: ${financial_metrics.get('Current Price', 0):,.2f}")
        st.write(f"- **Trend**: {trend}")
        st.write(f"- **Confidence**: {confidence:.2f}")
        st.write(f"- **Predicted Price (Next Day)**: ${predicted_price:.2f}" if predicted_price is not None else "- **Predicted Price (Next Day)**: N/A")
        st.write(f"- **Current Price**: ${financial_metrics.get('Current Price', 0):,.2f}")
    # Obtener fechas de vencimiento
    expiration_dates = get_expiration_dates(stock)
    if not expiration_dates:
        st.stop()

    # Seleccionar una fecha de vencimiento
    

    # Obtener datos de opciones para la fecha seleccionada
    option_data = get_option_data(stock, selected_expiration)
    if option_data.empty:
        st.stop()

    # Filtrar calls y puts
    calls = option_data[option_data["type"] == "call"]
    puts = option_data[option_data["type"] == "put"]

    # Crear una lista completa de strikes
    all_strikes = sorted(set(option_data["strike"]))

    # Rellenar datos faltantes con ceros
    calls_filled = pd.DataFrame({"strike": all_strikes}).merge(calls, on="strike", how="left").fillna(0)
    puts_filled = pd.DataFrame({"strike": all_strikes}).merge(puts, on="strike", how="left").fillna(0)

    # Crear el gráfico de histograma
    fig = go.Figure()

    # Añadir barras para CALLS (hacia arriba)
    for _, row in calls_filled.iterrows():
        fig.add_trace(
            go.Bar(
                name=f"Call ({'buy' if row['action'] == 'buy' else 'sell'})",
                x=[row["strike"]],
                y=[row["open_interest"]] if row["action"] == "buy" else [-row["open_interest"]],
                marker_color="green" if row["action"] == "buy" else "orange",
                text=f"{str(row['action']).capitalize()} Call",
                textposition="inside",
                orientation="v",
            )
        )

    # Añadir barras para PUTS (hacia abajo)
    for _, row in puts_filled.iterrows():
        fig.add_trace(
            go.Bar(
                name=f"Put ({'buy' if row['action'] == 'buy' else 'sell'})",
                x=[row["strike"]],
                y=[-row["open_interest"]] if row["action"] == "buy" else [row["open_interest"]],
                marker_color="red" if row["action"] == "buy" else "purple",
                text=f"{str(row['action']).capitalize()} Put",
                textposition="inside",
                orientation="v",
            )
        )

    # Personalizar el diseño del gráfico
    fig.update_layout(
        title=f"Calls and Puts for {stock} (Expiration: {selected_expiration})",
        xaxis_title="Strike Price",
        yaxis_title="SELLS/ PUTS & CALLS",
        barmode="relative",  # Barras relativas para superposición
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(range=[min(all_strikes) - 10, max(all_strikes) + 10]),  # Ajustar rango del eje X
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)
    st.write(f"- **Current Price**: ${financial_metrics.get('Current Price', 0):,.2f}")
    
    # Mostrar la tabla de datos con estilos dinámicos
    st.subheader("Data")
    st.write(f"- **Current Price**: ${financial_metrics.get('Current Price', 0):,.2f}")
    styled_option_data = style_option_data(option_data)
    st.dataframe(styled_option_data)  # Mostrar la tabla con estilos
    st.write(f"- **Current Price**: ${financial_metrics.get('Current Price', 0):,.2f}")
