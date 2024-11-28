import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import time
import yfinance as yf
import warnings
import bcrypt
import streamlit_authenticator as stauth
import csv


st.set_page_config(page_title="Ozy Target", layout="wide")



# Función para cargar usuarios desde un archivo CSV
def load_users():
    users = {}
    try:
        with open("users.csv", mode="r") as file:
            reader = csv.reader(file)
            for row in reader:
                email, hashed_password = row
                users[email] = {"password": hashed_password}
    except FileNotFoundError:
        # Crear el archivo si no existe
        with open("users.csv", mode="w", newline="") as file:
            pass
    return users

# Función para registrar un usuario
def register_user(email, password):
    try:
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        with open("users.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([email, hashed_password])
        return "Registro exitoso"
    except Exception as e:
        return f"Error al registrar el usuario: {e}"

# Función para autenticar al usuario
def authenticate_user(email, password):
    if email in users:
        hashed_password = users[email]["password"]
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    return False

# Cargar usuarios al iniciar la app
users = load_users()

# Inicializar el estado de sesión si no existe
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

# Función para recargar la página
def reload_page():
    st.session_state["reload"] = True

# Mostrar contenido basado en autenticación
if not st.session_state["authenticated"]:
    # Mostrar opciones de registro e inicio de sesión
    with st.sidebar.expander("Registro"):
        email_register = st.text_input("Correo electrónico", key="register_email")
        password_register = st.text_input("Contraseña", type="password", key="register_password")
        if st.button("Registrar"):
            if email_register and password_register:
                if email_register in users:
                    st.error("El correo ya está registrado.")
                else:
                    message = register_user(email_register, password_register)
                    users = load_users()  # Recargar usuarios
                    st.success(message)
            else:
                st.error("Por favor, completa ambos campos.")

    with st.sidebar.expander("Inicio de Sesión"):
        login_email = st.text_input("Correo electrónico", key="login_email")
        login_password = st.text_input("Contraseña", type="password", key="login_password")
        if st.button("Iniciar Sesión"):
            if login_email and login_password:
                if authenticate_user(login_email, login_password):
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = login_email
                    reload_page()  # Recargar página
                else:
                    st.sidebar.error("Credenciales incorrectas.")
            else:
                st.sidebar.error("Por favor, completa ambos campos.")
else:
    # Si está autenticado, ocultar registro e inicio de sesión y mostrar bienvenida
    st.sidebar.success(f"Bienvenido, {st.session_state['user_email']}!")

    # Botón para cerrar sesión
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state["authenticated"] = False
        st.session_state["user_email"] = None
        reload_page()  # Recargar página

# Proteger el contenido principal
if not st.session_state["authenticated"]:
    st.warning("Por favor, inicia sesión para acceder a las herramientas.")
    st.stop()



#AQUI COMIENZA LA APP>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# Configuración de la API Tradier
API_KEY = "U1iAJk1HhOCfHxULqzo2ywM2jUAX"
BASE_URL = "https://api.tradier.com/v1"

# Configuración de la página

st.title("Advanced Tools")
  
UPDATE_INTERVAL = 15









@st.cache_data
def analyze_advanced_potential_profit(strikes_data, current_price, time_to_expiration):
    advanced_contracts = []
    if not strikes_data:  # Verifica si los datos están vacíos
        return pd.DataFrame(columns=["Type", "Strike", "Volume", "OI", "Potential Profit (%)", "Advanced Score (%)"])

    for strike, data in strikes_data.items():
        for option_type in ["CALL", "PUT"]:
            # Extraer datos básicos
            volume = data[option_type].get("VOLUME", 0)
            oi = data[option_type].get("OI", 0)
            delta = abs(data[option_type].get("DELTA", 0))  # Delta absoluto
            iv = data[option_type].get("IV", 0)  # Volatilidad implícita
            theta = abs(data[option_type].get("THETA", 0))  # Theta absoluto

            # Calcular la ganancia potencial estándar
            if option_type == "CALL" and strike > current_price:
                potential_profit = ((strike - current_price) / current_price) * 100
            elif option_type == "PUT" and strike < current_price:
                potential_profit = ((current_price - strike) / current_price) * 100
            else:
                continue  # Omitir opciones irrelevantes

            # Calcular puntaje avanzado
            if time_to_expiration > 0:
                time_factor = 1 + (time_to_expiration / (time_to_expiration + 10))  # Factor de tiempo ajustado
            else:
                time_factor = 1

            advanced_score = potential_profit * (1 + (delta * 0.4) - (iv * 0.3) + (time_factor * 0.3))

            # Filtrar contratos con alta ganancia potencial (>60%) y un buen puntaje avanzado
            if potential_profit > 60 or advanced_score > 60:
                advanced_contracts.append({
                    "Type": option_type,
                    "Strike": strike,
                    "Volume": volume,
                    "OI": oi,
                    "Potential Profit (%)": potential_profit,
                    "Advanced Score (%)": advanced_score
                })

    return pd.DataFrame(advanced_contracts) if advanced_contracts else pd.DataFrame(columns=["Type", "Strike", "Volume", "OI", "Potential Profit (%)", "Advanced Score (%)"])


# Gráfico avanzado para el puntaje
def plot_advanced_scores(contracts_df):
    if contracts_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contracts_df["Strike"],
        y=contracts_df["Advanced Score (%)"],
        name="Advanced Score (%)",
        marker_color="green"
    ))
    fig.update_layout(
        title="Advanced Score by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Advanced Score (%)",
        template="plotly_white"
    )
    return fig










# Función para calcular ganancias potenciales de contratos
def analyze_potential_profit(strikes_data, current_price):
    potential_contracts = []

    for strike, data in strikes_data.items():
        # Calcular la ganancia potencial para CALL
        if strike > current_price and "CALL" in data:
            potential_profit = ((strike - current_price) / current_price) * 100
            if potential_profit > 60:
                potential_contracts.append({
                    "Type": "CALL",
                    "Strike": strike,
                    "Volume": data["CALL"].get("VOLUME", 0),
                    "OI": data["CALL"].get("OI", 0),
                    "Potential Profit (%)": round(potential_profit, 2)
                })

        # Calcular la ganancia potencial para PUT
        if strike < current_price and "PUT" in data:
            potential_profit = ((current_price - strike) / current_price) * 100
            if potential_profit > 60:
                potential_contracts.append({
                    "Type": "PUT",
                    "Strike": strike,
                    "Volume": data["PUT"].get("VOLUME", 0),
                    "OI": data["PUT"].get("OI", 0),
                    "Potential Profit (%)": round(potential_profit, 2)
                })

    # Convertir a DataFrame
    return pd.DataFrame(potential_contracts) if potential_contracts else pd.DataFrame(columns=["Type", "Strike", "Volume", "OI", "Potential Profit (%)"])









# Ignorar advertencias deprecadas
warnings.filterwarnings("ignore", category=DeprecationWarning)
@st.cache_data
def get_quarterly_eps(ticker):
    try:
        stock = yf.Ticker(ticker)

        # Obtener el estado de resultados trimestral
        quarterly_financials = stock.quarterly_financials
        if quarterly_financials.empty:
            return None

        # Obtener Net Income y Shares Outstanding
        net_income = quarterly_financials.loc["Net Income"].values  # Ingresos netos
        shares_outstanding = stock.info.get("sharesOutstanding")  # Acciones en circulación
        quarters = quarterly_financials.columns.astype(str)  # Fechas de los trimestres

        if shares_outstanding is None or len(net_income) == 0:
            return None

        # Calcular EPS trimestral
        eps_values = net_income / shares_outstanding

        # Obtener EPS estimado para trimestres futuros
        try:
            future_estimates = stock.analysis["Earnings Estimate"]  # Datos de estimaciones futuras
            estimated_eps = future_estimates.iloc[1:3, 1].values  # Tomar las próximas dos estimaciones trimestrales
            future_quarters = ["2024-12-31", "2025-03-31"]  # Trimestres futuros
        except Exception:
            estimated_eps = [None, None]
            future_quarters = ["2024-12-31", "2025-03-31"]

        # Crear DataFrame con los datos
        eps_data = pd.DataFrame({
            "Quarter": list(quarters) + future_quarters,
            "Net Income": list(net_income) + [None] * len(future_quarters),
            "Shares Outstanding": [shares_outstanding] * len(quarters) + [None] * len(future_quarters),
            "EPS": list(eps_values) + [None] * len(future_quarters),
            "Estimated EPS": [None] * len(quarters) + list(estimated_eps)
        })

        # Ordenar por fechas
        eps_data["Quarter"] = pd.to_datetime(eps_data["Quarter"])
        eps_data.sort_values(by="Quarter", inplace=True)
        eps_data["Quarter"] = eps_data["Quarter"].dt.strftime("%Y-%m-%d")  # Convertir de nuevo a texto
        return eps_data
    except Exception as e:
        st.error(f"Error fetching EPS data: {e}")
        return None

# Crear gráfico de EPS trimestral
def create_eps_chart(eps_data):
    fig = go.Figure()

    # Línea de EPS real
    fig.add_trace(go.Scatter(
        x=eps_data["Quarter"],
        y=eps_data["EPS"],
        mode="markers+lines",
        name="Actual EPS",
        marker=dict(color="orange", size=10),
        line=dict(color="orange", width=2)
    ))

    # Línea de EPS estimado
    fig.add_trace(go.Scatter(
        x=eps_data["Quarter"],
        y=eps_data["Estimated EPS"],
        mode="markers+lines",
        name="Estimated EPS",
        marker=dict(color="brown", size=10),
        line=dict(color="brown", width=2, dash="dash")
    ))

    # Configuración del diseño del gráfico
    fig.update_layout(
        title="EPS Analysis (Actual vs Estimated - Quarterly)",
        xaxis=dict(title="Quarter"),
        yaxis=dict(title="EPS"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        template="plotly_dark",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000"
    )
    return fig




@st.cache_data
def find_best_contracts(strikes_data):
    best_contracts = []
    if not strikes_data:  # Verifica si está vacío
        return pd.DataFrame(columns=["Type", "Strike", "Volume", "OI", "Activity (%)"])
    
    for strike, data in strikes_data.items():
        call_volume = data["CALL"].get("VOLUME", 0)
        put_volume = data["PUT"].get("VOLUME", 0)
        call_oi = data["CALL"].get("OI", 0)
        put_oi = data["PUT"].get("OI", 0)

        # Calcular actividad inusual
        call_activity = (call_volume / call_oi * 100) if call_oi > 0 else 0
        put_activity = (put_volume / put_oi * 100) if put_oi > 0 else 0

        # Agregar contratos relevantes
        if call_activity > 50:
            best_contracts.append({
                "Type": "CALL",
                "Strike": strike,
                "Volume": call_volume,
                "OI": call_oi,
                "Activity (%)": call_activity
            })
        if put_activity > 50:
            best_contracts.append({
                "Type": "PUT",
                "Strike": strike,
                "Volume": put_volume,
                "OI": put_oi,
                "Activity (%)": put_activity
            })

    return pd.DataFrame(best_contracts) if best_contracts else pd.DataFrame(columns=["Type", "Strike", "Volume", "OI", "Activity (%)"])








# Función ajustada para calcular el porcentaje de volumen entre CALL y PUT
def calculate_ticker_call_put_percentage(data):
    if not data or len(data) == 0:
        return {"Calls (%)": 0, "Puts (%)": 0}

    total_calls = 0
    total_puts = 0

    # Navegar por las claves del strike 
    for strike, strike_data in data.items():
        if "CALL" in strike_data and "VOLUME" in strike_data["CALL"]:
            total_calls += strike_data["CALL"]["VOLUME"]
        if "PUT" in strike_data and "VOLUME" in strike_data["PUT"]:
            total_puts += strike_data["PUT"]["VOLUME"]

    total_volume = total_calls + total_puts

    if total_volume == 0:
        return {"Calls (%)": 0, "Puts (%)": 0}

    return {
        "Calls (%)": (total_calls / total_volume) * 100,
        "Puts (%)": (total_puts / total_volume) * 100,
    }


@st.cache_data
def get_price_data(ticker):
    url = f"{BASE_URL}/markets/quotes"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbols": ticker}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("quotes", {}).get("quote", {})
    else:
        st.error("Error fetching data from Tradier API")
        return None

# Función para obtener fechas de expiración
@st.cache_data
def get_expiration_dates(ticker):
    url = f"{BASE_URL}/markets/options/expirations"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get("expirations", {}).get("date", [])
    else:
        st.error("Error fetching expiration dates")
        return []

# Función para obtener datos de opciones, incluyendo Delta y Theta
@st.cache_data
def get_options_data(ticker, expiration_date):
    url = f"{BASE_URL}/markets/options/chains"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker, "expiration": expiration_date, "greeks": "true"}  # Activar greeks
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        options = response.json().get("options", {}).get("option", [])
        strikes_data = {}
        for option in options:
            strike = option["strike"]
            option_type = option["option_type"]
            open_interest = option.get("open_interest", 0)
            volume = option.get("volume", 0)
            delta = option.get("greeks", {}).get("delta", 0)  # Extraer Delta
            theta = option.get("greeks", {}).get("theta", 0)  # Extraer Theta
            if strike not in strikes_data:
                strikes_data[strike] = {"CALL": {"OI": 0, "VOLUME": 0, "DELTA": 0, "THETA": 0},
                                        "PUT": {"OI": 0, "VOLUME": 0, "DELTA": 0, "THETA": 0}}
            if option_type == "call":
                strikes_data[strike]["CALL"]["OI"] += open_interest
                strikes_data[strike]["CALL"]["VOLUME"] += volume
                strikes_data[strike]["CALL"]["DELTA"] = delta
                strikes_data[strike]["CALL"]["THETA"] = theta
            elif option_type == "put":
                strikes_data[strike]["PUT"]["OI"] += open_interest
                strikes_data[strike]["PUT"]["VOLUME"] += volume
                strikes_data[strike]["PUT"]["DELTA"] = delta
                strikes_data[strike]["PUT"]["THETA"] = theta
        return strikes_data
    else:
        st.error("Error fetching options data")
        return {}



def execute_trade(api_key, contract_type, strike, volume):
    url = f"https://api.tradier.com/v1/accounts/<account_id>/orders"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    payload = {
        "class": "option",
        "symbol": ticker,
        "option_type": contract_type.lower(),
        "strike": strike,
        "quantity": volume,
        "price": "market",
        "side": "buy",  # or "sell"
        "duration": "day",
        "type": "market"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        st.success(f"Trade executed: {contract_type} at ${strike}")
    else:
        st.error(f"Trade failed: {response.json().get('message')}")


def plot_contract_activity(contracts_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=contracts_df["Strike"],
        y=contracts_df["Volume"],
        name="Volume",
        marker_color="blue"
    ))
    fig.add_trace(go.Bar(
        x=contracts_df["Strike"],
        y=contracts_df["OI"],
        name="Open Interest",
        marker_color="orange"
    ))
    fig.update_layout(
        title="Volume vs Open Interest by Strike",
        xaxis_title="Strike Price",
        yaxis_title="Contracts",
        barmode="group",
        template="plotly_white"
    )
    return fig

def plot_contract_activity(best_contracts):
    if best_contracts.empty:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=best_contracts["Strike"],
        y=best_contracts["Activity (%)"],
        name="Activity (%)",
        marker_color="blue"
    ))
    fig.update_layout(
        title="Volume vs Open Interest Activity",
        xaxis_title="Strike Price",
        yaxis_title="Activity (%)",
        template="plotly_white"
    )
    return fig








# Función mejorada para calcular Max Pain
def calculate_advanced_max_pain(strikes_data, metric, time_to_expiration=1):
    max_pain_values = {}
    total_open_interest = sum(
        data["CALL"][metric] + data["PUT"][metric] for data in strikes_data.values()
    )

    for target_strike in sorted(strikes_data.keys()):
        total_pain = 0
        for strike, data in strikes_data.items():
            distance = abs(target_strike - strike)
            time_weight = 1 / (time_to_expiration + 1)  # Peso por tiempo a expiración
            delta_weight = abs(data["CALL"].get("DELTA", 0.5)) + abs(data["PUT"].get("DELTA", 0.5))  # Ponderación por delta
            weight = (1 / (distance + 1)) * time_weight * delta_weight

            call_pain = data["CALL"][metric] * (target_strike - strike) * weight
            put_pain = data["PUT"][metric] * (strike - target_strike) * weight
            total_pain += call_pain + put_pain

        max_pain_values[target_strike] = total_pain / total_open_interest if total_open_interest else total_pain

    return min(max_pain_values, key=max_pain_values.get)



# Función para obtener los top N strikes por métrica, incluyendo Delta y Theta
def get_top_strikes_with_greeks(strikes_data, metric, option_type, top_n=4):
    strikes = []
    for strike, data in strikes_data.items():
        value = data[option_type][metric]
        delta = data[option_type].get("DELTA", 0)  # Delta por strike
        theta = data[option_type].get("THETA", 0)  # Theta por strike
        strikes.append({"Strike": strike, metric: value, "Delta": delta, "Theta": theta})
    df = pd.DataFrame(strikes).sort_values(by=metric, ascending=False).head(top_n)
    return df

# Función para crear el gráfico de Gamma Exposure
def gamma_exposure_chart(strikes_data, current_price):
    strikes = sorted(strikes_data.keys())
    gamma_calls = [strikes_data[s]["CALL"]["OI"] for s in strikes]
    gamma_puts = [strikes_data[s]["PUT"]["OI"] for s in strikes]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=strikes, y=gamma_calls, name="Gamma CALL", marker_color="blue"))
    fig.add_trace(go.Bar(x=strikes, y=[-g for g in gamma_puts], name="Gamma PUT", marker_color="red"))

    # Línea para el precio actual
    fig.add_shape(
        type="line",
        x0=current_price, x1=current_price,
        y0=min(-max(gamma_puts), -max(gamma_calls)),
        y1=max(gamma_calls),
        line=dict(color="orange", width=1, dash="dot")
    )
    fig.add_annotation(
        x=current_price,
        y=max(gamma_calls) * 0.9,
        text=f"Current Price: ${current_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="orange",
        font=dict(color="orange", size=12)
    )

    fig.update_layout(
        title="Gamma Exposure",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure",
        barmode="relative",
        template="plotly_white"
    )
    return fig
    
@st.cache_data
def scan_top_5_precise_moving_stocks():
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
        "AMD", "BA", "ORCL", "INTC", "PYPL", "CSCO", "ADBE", "AVGO", "CRM",
        "TXN", "QCOM", "INTU", "SHOP", "AMAT", "AMD", "V", "MA", "JNJ", "PFE",
        "MRNA", "WMT", "TGT", "COST", "HD", "LOW", "DIS", "NKE", "SBUX", "PEP",
        "KO", "XOM", "CVX", "BP", "COP", "SPY", "QQQ", "DIA", "UNH", "ABBV",
        "TMO", "LIN", "HON", "IWM", "BITX", "ENPH", "YINN", "NEE", "SMCI",
        "BAC", "GGAL", "MARA", "COIN", "ROOT", "NU"
    ]

    url = f"{BASE_URL}/markets/quotes"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbols": ",".join(symbols)}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        quotes = response.json().get("quotes", {}).get("quote", [])
        if isinstance(quotes, dict):  # Si Tradier devuelve un único resultado
            quotes = [quotes]

        stocks = []
        for stock in quotes:
            symbol = stock.get("symbol", "")
            last_price = stock.get("last", 0)
            change_percent = abs(stock.get("change_percentage", 0))
            volume = stock.get("volume", 0)
            avg_volume = stock.get("average_volume", 1)
            rel_volume = volume / avg_volume if avg_volume > 0 else 0
            high = stock.get("high", last_price)
            low = stock.get("low", last_price)
            spread_percent = ((high - low) / last_price) * 100 if last_price > 0 else 0
            iv = stock.get("volatility", 0)

            if iv == 0:
                projected_move = (rel_volume * 0.4) + (change_percent * 0.4) + (spread_percent * 0.2)
            else:
                projected_move = (rel_volume * 0.25) + (change_percent * 0.25) + (iv * 0.3) + (spread_percent * 0.2)

            stocks.append({
                "Symbol": symbol,
                "Price": last_price,
                "Change (%)": change_percent,
                "Volume": volume,
                "Relative Volume": rel_volume,
                "Spread (%)": spread_percent,
                "IV": iv,
                "Projected Move (%)": projected_move
            })

        return pd.DataFrame(stocks).sort_values(by="Projected Move (%)", ascending=False).head(5)
    else:
        st.error(f"Error fetching market data: {response.text}")
        return pd.DataFrame()

# Función para graficar el movimiento proyectado
def projected_movement_chart(stocks_df):
    fig = go.Figure(data=[
        go.Bar(name="Projected Move (%)", x=stocks_df["Symbol"], y=stocks_df["Projected Move (%)"])
    ])
    fig.update_layout(
        title="Top 5 Projected ",
        xaxis_title="Symbols",
        yaxis_title="Projected Movement (%)",
        template="plotly_white"
    )
    return fig

# Escaneo de las Magníficas 50
with st.expander("Top  Moving Stocks"):
    st.subheader("")
    top_5_stocks = scan_top_5_precise_moving_stocks()
    if not top_5_stocks.empty:
        st.dataframe(top_5_stocks)
        st.plotly_chart(projected_movement_chart(top_5_stocks), use_container_width=True)
    else:
        st.write("No data available for the selected stocks.")


        # Función para crear un gráfico de pastel para visualizar el porcentaje
def ticker_call_put_pie_chart(percentages):
    labels = ['Calls', 'Puts']
    values = [percentages['Calls (%)'], percentages['Puts (%)']]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(
        title="Call vs Put Volume Percentage for Selected Ticker",
        template="plotly_white"
    )
    return fig












# Sidebar para entradas
st.sidebar.subheader("Inputs")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper()
# Contenedores dinámicos
price_container = st.empty()
metrics_container = st.empty()
gamma_chart_container = st.empty()

# Sección 1: Información del Precio Actual

if ticker:
    expiration_dates = get_expiration_dates(ticker)
    if expiration_dates:
        expiration_date = st.sidebar.selectbox("Expiration", expiration_dates)
        price_data = get_price_data(ticker)

        if price_data:
            with st.expander("Current "):
                st.write(f"**Last Price:** ${price_data['last']}")
                st.write(f"**High:** ${price_data['high']}")
                st.write(f"**Low:** ${price_data['low']}")
                st.write(f"**Volume:** {price_data['volume']}")

        strikes_data = get_options_data(ticker, expiration_date)
      
        # Sección 2: Métricas Clave
        if strikes_data:
            time_to_expiration = max(1, (pd.to_datetime(expiration_date) - pd.Timestamp.today()).days)

            gamma_max_pain = calculate_advanced_max_pain(strikes_data, "OI", time_to_expiration=5)
            volume_max_pain = calculate_advanced_max_pain(strikes_data, "VOLUME", time_to_expiration=5)


            with st.expander("Target Expiration"):
                st.write(f"**Target -:** ${gamma_max_pain}")
                st.write(f"**Target:** ${volume_max_pain}")
                

            # Sección 3: Top Strikes con Delta y Theta
            with st.expander("Options Strikes"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("####  4 CALL Interest ")
                    top_call_oi = get_top_strikes_with_greeks(strikes_data, "OI", "CALL")
                    st.dataframe(top_call_oi)

                    st.markdown("####  CALL Volume ")
                    top_call_volume = get_top_strikes_with_greeks(strikes_data, "VOLUME", "CALL")
                    st.dataframe(top_call_volume)

                with col2:
                    st.markdown("#### PUT  Interest")
                    top_put_oi = get_top_strikes_with_greeks(strikes_data, "OI", "PUT")
                    st.dataframe(top_put_oi)

                    st.markdown("#### PUT  Volume")
                    top_put_volume = get_top_strikes_with_greeks(strikes_data, "VOLUME", "PUT")
                    st.dataframe(top_put_volume)

            # Sección 4: Gráfico de Gamma Exposure
            current_price = price_data["last"]
            with st.expander(" Gamma  Chart"):
                gamma_chart = gamma_exposure_chart(strikes_data, current_price)
                st.plotly_chart(gamma_chart, use_container_width=True)
                time.sleep(UPDATE_INTERVAL)
                # Obtener datos del mercado


               

# Mejoras en el desglose de Call/Put %
with st.expander("Call / Put % Analysis"):
    st.subheader("Call vs Put Volume Percentage for Selected Ticker")

    if ticker and expiration_date:
        # Obtener datos de opciones para el ticker
        ticker_options = get_options_data(ticker, expiration_date)

        # Mostrar datos en formato JSON si se activa la opción
        if st.checkbox("Show Raw Options Data"):
            st.json(ticker_options)

        if ticker_options:
            # Calcular porcentajes de volumen de Call/Put
            percentages = calculate_ticker_call_put_percentage(ticker_options)

            # Mostrar resultados
            st.write(f"**Calls (%):** {percentages['Calls (%)']:.2f}%")
            st.write(f"**Puts (%):** {percentages['Puts (%)']:.2f}%")

            # Gráfico de pastel
            st.plotly_chart(ticker_call_put_pie_chart(percentages), use_container_width=True)
        else:
            st.warning("No options data available for the selected ticker and expiration date.")
    else:
        st.warning("Please enter a valid ticker and expiration date.")

# Mejoras en el análisis de Best Contracts
with st.expander("Contracts Analysis"):
    st.subheader("Contracts with Unusual Activity")

    if strikes_data:
        # Encontrar los mejores contratos
        best_contracts = find_best_contracts(strikes_data)

        if not best_contracts.empty:
            st.write("### Top Contracts with High Activity")
            st.dataframe(best_contracts)

            # Gráfico de contratos
            st.plotly_chart(plot_contract_activity(best_contracts), use_container_width=True)
        else:
            st.warning("No contracts with significant activity found.")
    else:
        st.warning("Options data is not available for this ticker.")






# Interfaz de Streamlit


if ticker:
    with st.expander("📊 EPS Quarterly Analysis"):
        eps_data = get_quarterly_eps(ticker)

        if eps_data is not None:
            # Mostrar los datos en una tabla
            st.write("### EPS Data Table (Quarterly)")
            st.dataframe(eps_data)

            # Mostrar el gráfico
            st.plotly_chart(create_eps_chart(eps_data), use_container_width=True)
        else:
            st.warning(f"No EPS data available for {ticker}.")




# Integrar análisis de contratos con más del 60% de ganancia
with st.expander("Analyze Profitable Contracts"):
    st.subheader("Identify contracts with over 60% potential profit.")
    
    if strikes_data and price_data:
        # Obtener el precio actual
        current_price = price_data["last"]

        # Realizar el análisis
        profitable_contracts = analyze_potential_profit(strikes_data, current_price)

        if not profitable_contracts.empty:
            st.write("### Contracts with Over 60% Potential Profit")
            st.dataframe(profitable_contracts)

            # Opcional: Crear un gráfico
            fig = go.Figure(data=[
                go.Bar(
                    x=profitable_contracts["Strike"],
                    y=profitable_contracts["Potential Profit (%)"],
                    name="Potential Profit (%)"
                )
            ])
            fig.update_layout(
                title="Potential Profit by Strike",
                xaxis_title="Strike Price",
                yaxis_title="Potential Profit (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No contracts found with more than 60% potential profit.")
    else:
        st.warning("Options data is not available for this ticker.")







# Sección en Streamlit para contratos avanzados
with st.expander("Analyze Advanced Profitable Contracts"):
    st.subheader("Identify contracts with advanced scoring for high potential profit.")
    
    if strikes_data and expiration_date:
        # Calcular días hasta la expiración
        time_to_expiration = max(1, (pd.to_datetime(expiration_date) - pd.Timestamp.today()).days)

        # Calcular contratos con la fórmula avanzada
        advanced_results = analyze_advanced_potential_profit(strikes_data, price_data['last'], time_to_expiration)

        if not advanced_results.empty:
            st.write("### Contracts with High Advanced Score")
            st.dataframe(advanced_results)

            # Mostrar el gráfico
            st.plotly_chart(plot_advanced_scores(advanced_results), use_container_width=True)
        else:
            st.write("No contracts with significant advanced score found.")
    else:
        st.warning("Options data or expiration date is not available.")
