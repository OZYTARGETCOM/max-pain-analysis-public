import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import time
import yfinance as yf
import warnings
import bcrypt
import streamlit_authenticator as stauth
import csv

# Configuración inicial de la página
st.set_page_config(page_title="SCANNER OPTIONS", layout="wide")

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>seguridad




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
    # Mostrar opciones de registro e inicio de sesión en la pantalla principal
    st.header("🔑 Ingreso y Registro")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Registro")
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

    with col2:
        st.subheader("Inicio de Sesión")
        login_email = st.text_input("Correo electrónico", key="login_email")
        login_password = st.text_input("Contraseña", type="password", key="login_password")
        if st.button("Iniciar Sesión"):
            if login_email and login_password:
                if authenticate_user(login_email, login_password):
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = login_email
                    reload_page()  # Recargar página
                else:
                    st.error("Credenciales incorrectas.")
            else:
                st.error("Por favor, completa ambos campos.")
else:
    # Si está autenticado, mostrar bienvenida en la pantalla principal
    st.success(f"Bienvenido, {st.session_state['user_email']}!")

    # Botón para cerrar sesión en la pantalla principal
    if st.button("Cerrar Sesión"):
        st.session_state["authenticated"] = False
        st.session_state["user_email"] = None
        reload_page()  # Recargar página

# Proteger el contenido principal
if not st.session_state["authenticated"]:
    st.warning("Por favor, inicia sesión para acceder a las herramientas.")
    st.stop()

    
#BUENOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

















#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>end seguridad    














# Configuración de la API Tradier
API_KEY = "U1iAJk1HhOCfHxULqzo2ywM2jUAX"
BASE_URL = "https://api.tradier.com/v1"

# Función para obtener datos de opciones
@st.cache_data
def get_options_data(ticker, expiration_date):
    url = f"{BASE_URL}/markets/options/chains"
    headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
    params = {"symbol": ticker, "expiration": expiration_date, "greeks": "true"}  # Activar Greeks
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json().get("options", {}).get("option", [])
    else:
        st.error("Error fetching options data.")
        return []

# Función para obtener el precio actual
@st.cache_data
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
        st.error("Error fetching expiration dates.")
        return []

# Función para calcular Max Pain
def calculate_max_pain(options_data):
    if not options_data:
        return None
    strikes = {}
    for option in options_data:
        strike = option["strike"]
        open_interest = option["open_interest"] or 0
        if strike not in strikes:
            strikes[strike] = 0
        strikes[strike] += open_interest
    return max(strikes, key=strikes.get)

# Función para calcular puntuación avanzada
def calculate_advanced_score(option, current_price, max_pain):
    open_interest = option["open_interest"] or 0
    volume = option["volume"] or 0
    implied_volatility = option.get("implied_volatility", 0)
    strike = option["strike"]

    # Calcular puntuación basada en los factores
    score = (0.4 * open_interest) + (0.3 * volume) + (0.2 * implied_volatility) - (0.1 * abs(max_pain - strike))
    return score

# Función para seleccionar los mejores contratos
def select_best_contracts(options_data, current_price):
    max_pain = calculate_max_pain(options_data)
    best_contracts = []

    for option in options_data:
        score = calculate_advanced_score(option, current_price, max_pain)
        entry_price = option["last"] or option["bid"] or 0

        # Filtrar contratos razonables (no precios irreales)
        if score > 0 and entry_price > 0 and abs(option["strike"] - current_price) <= 20:
            option["score"] = score
            best_contracts.append(option)

    # Ordenar por puntuación y seleccionar los top 6
    best_contracts = sorted(best_contracts, key=lambda x: x["score"], reverse=True)[:6]
    return best_contracts


import plotly.graph_objects as go

def gamma_exposure_chart(strikes_data, current_price):
    # Ordenar los strikes
    strikes = sorted(strikes_data.keys())

    # Extraer datos de Gamma para Calls y Puts
    gamma_calls = [strikes_data[s]["CALL"]["OI"] * strikes_data[s]["CALL"]["Gamma"] for s in strikes]
    gamma_puts = [strikes_data[s]["PUT"]["OI"] * strikes_data[s]["PUT"]["Gamma"] for s in strikes]

    # Crear la figura
    fig = go.Figure()

    # Barras de Gamma Exposure para Calls
    fig.add_trace(go.Bar(
        x=strikes,
        y=gamma_calls,
        name="Gamma CALL",
        marker_color="blue"
    ))

    # Barras de Gamma Exposure para Puts (invertido)
    fig.add_trace(go.Bar(
        x=strikes,
        y=[-g for g in gamma_puts],
        name="Gamma PUT",
        marker_color="red"
    ))

    # Añadir línea para el precio actual
    fig.add_shape(
        type="line",
        x0=current_price,
        x1=current_price,
        y0=min(-max(gamma_puts), min(gamma_calls)) * 1.1,
        y1=max(max(gamma_calls), -min(gamma_puts)) * 1.1,
        line=dict(color="orange", width=2, dash="dot")
    )

    # Añadir anotación para el precio actual
    fig.add_annotation(
        x=current_price,
        y=max(max(gamma_calls), -min(gamma_puts)) * 0.9,
        text=f"Current Price: ${current_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="orange",
        font=dict(color="orange", size=12)
    )

    # Actualizar diseño del gráfico
    fig.update_layout(
        title="Gamma Exposure (Calls vs Puts)",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure",
        barmode="relative",
        template="plotly_white",
        legend=dict(title="Option Type")
    )

    return fig











#grafica de precio de contrato  
def risk_return_chart_auto(strike_price, premium_paid, current_price, contract_type):
    # Rango de precios del subyacente alrededor del precio actual (±20%)
    prices = np.linspace(current_price * 0.8, current_price * 1.2, 100)
    profit_loss = []

    # Calcular ganancia/pérdida para cada precio del subyacente
    if contract_type.upper() == "CALL":
        profit_loss = [(max(price - strike_price, 0) - premium_paid) for price in prices]
    elif contract_type.upper() == "PUT":
        profit_loss = [(max(strike_price - price, 0) - premium_paid) for price in prices]
    else:
        raise ValueError("Invalid contract type. Use 'CALL' or 'PUT'.")

    # Crear figura del gráfico
    fig = go.Figure()

    # Añadir la línea de ganancia/pérdida
    fig.add_trace(go.Scatter(
        x=prices,
        y=profit_loss,
        mode='lines',
        name='Profit/Loss',
        line=dict(color="green")
    ))

    # Línea para el precio actual
    fig.add_shape(
        type="line",
        x0=current_price,
        x1=current_price,
        y0=min(profit_loss),
        y1=max(profit_loss),
        line=dict(color="orange", width=2, dash="dot"),
        name="Current Price"
    )

    # Líneas para el precio de ejercicio (strike)
    fig.add_shape(
        type="line",
        x0=strike_price,
        x1=strike_price,
        y0=min(profit_loss),
        y1=max(profit_loss),
        line=dict(color="blue", width=2, dash="dot"),
        name="Strike Price"
    )

    # Anotaciones para precio actual y strike
    fig.add_annotation(
        x=current_price,
        y=0,
        text=f"Current Price: ${current_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="orange",
        font=dict(color="orange", size=12)
    )
    fig.add_annotation(
        x=strike_price,
        y=0,
        text=f"Strike Price: ${strike_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="blue",
        font=dict(color="blue", size=12)
    )

    # Configurar el diseño del gráfico
    fig.update_layout(
        title=f"Risk/Return Chart for {contract_type.upper()}",
        xaxis_title="Underlying Price",
        yaxis_title="Profit/Loss ($)",
        template="plotly_white",
        legend=dict(title="Legend"),
        shapes=[
            dict(
                type="line",
                x0=strike_price,
                x1=strike_price,
                y0=min(profit_loss),
                y1=max(profit_loss),
                line=dict(color="blue", width=2, dash="dot")
            )
        ]
    )

    return fig








# --- Función para generar recomendaciones basadas en IV, HV y tipo de contrato ---
def recommend_trades_based_on_iv_hv(options_data, historical_volatility):
    recommendations = []

    for option in options_data:
        strike = option["strike"]
        iv = option.get("implied_volatility", 0) * 100  # Convertir a porcentaje
        delta = option.get("greeks", {}).get("delta", 0)
        option_type = option["option_type"].upper()

        # Calcular la diferencia IV - HV
        iv_minus_hv = iv - historical_volatility

        # Reglas de recomendación para Calls
        if option_type == "CALL":
            if iv_minus_hv > 20 and delta > 0.8:
                recommendation = "Sell Call"
            elif iv_minus_hv < -10 and delta < 0.2:
                recommendation = "Buy Call"
            else:
                recommendation = "Avoid Call"

        # Reglas de recomendación para Puts
        elif option_type == "PUT":
            if iv_minus_hv > 20 and delta < -0.8:
                recommendation = "Sell Put"
            elif iv_minus_hv < -10 and delta > -0.2:
                recommendation = "Buy Put"
            else:
                recommendation = "Avoid Put"

        # Añadir datos relevantes
        recommendations.append({
            "Strike": strike,
            "Type": option_type,
            "IV": round(iv, 2),
            "HV": round(historical_volatility, 2),
            "IV - HV": round(iv_minus_hv, 2),
            "Delta": round(delta, 4),
            "Recommendation": recommendation
        })

    return recommendations

























# Función para formatear datos de contratos
def format_option_data(option, expiration_date, ticker, current_price):
    expiration_date_formatted = datetime.strptime(expiration_date, "%Y-%m-%d").strftime("%b-%d").upper()
    option_type = option["option_type"].upper()
    strike = option["strike"]
    entry_price = option["last"] or option["bid"] or 0

    # Cálculo ajustado de Max Gain basado en el movimiento esperado
    if option_type == "CALL":
        max_gain = ((strike - current_price) / entry_price * 100) if strike > current_price else -100
    elif option_type == "PUT":
        max_gain = ((current_price - strike) / entry_price * 100) if strike < current_price else -100
    else:
        max_gain = 0

    # Cálculo del riesgo-recompensa
    rr_ratio = max_gain / entry_price if entry_price > 0 else 0

    # Ajuste de valores de salida
    target_1 = round(entry_price * 1.1, 2)
    target_2 = round(entry_price * 1.2, 2)
    target_3 = round(entry_price * 1.3, 2)

    # Añadir valores de Greeks formateados
    delta = round(option.get("greeks", {}).get("delta", 0), 4)
    gamma = round(option.get("greeks", {}).get("gamma", 0), 4)
    theta = round(option.get("greeks", {}).get("theta", 0), 4)

    return {
        "Ticker": f"{ticker} {strike:.1f}/{option_type} {expiration_date_formatted}",
        "Entry": round(entry_price, 2),
        "Entry Date": datetime.now().strftime("%Y-%m-%d"),
        "Max Gain": f"{max_gain:.2f}%",
        "Risk-Reward Ratio": f"{rr_ratio:.2f}",
        "Status": "Live" if max_gain > 0 else "Closed at loss",
        "1st Exit": target_1,
        "2nd Exit": target_2,
        "3rd Exit": target_3,
        "Logo": "📈" if option_type == "CALL" else "📉",
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Bid": round(option["bid"], 2) if option["bid"] else "N/A",
        "Ask": round(option["ask"], 2) if option["ask"] else "N/A",
        "Volume": option["volume"] or "N/A",
        "Open Interest": option["open_interest"] or "N/A"
    }

# Interfaz de usuario
st.title("📊 Analyst & AI Signals")



ticker = st.text_input("Enter Ticker", value="SPY").upper()
expiration_dates = get_expiration_dates(ticker)
if expiration_dates:
    expiration_date = st.selectbox("Select Expiration Date", expiration_dates)
else:
    st.error("No expiration dates available.")
    st.stop()

current_price = get_current_price(ticker)
st.write(f"**Current Price for {ticker}:** ${current_price:.2f}")

options_data = get_options_data(ticker, expiration_date)
best_contracts = select_best_contracts(options_data, current_price)

st.subheader("Recommended Contracts")
for contract in best_contracts:
    formatted_data = format_option_data(contract, expiration_date, ticker, current_price)
    col1, col2, col3 = st.columns([1, 5, 2])

    # Cambiar colores dinámicamente según valores
    max_gain_color = "#28a745" if float(formatted_data["Max Gain"].strip('%')) > 0 else "#dc3545"
    rr_ratio_color = "#ffc107" if float(formatted_data["Risk-Reward Ratio"]) > 50 else "#dc3545"
    status_color = "#28a745" if formatted_data["Status"] == "Live" else "#dc3545"

    with col1:
        st.markdown(f"<h1 style='font-size:50px;'>{formatted_data['Logo']}</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="padding:10px; border:1px solid #ddd; border-radius:10px;">
            <h4 style="margin:0; color: #007bff;">{formatted_data['Ticker']}</h4>
            <p style="margin:0; color: #666;">Entry: {formatted_data['Entry']} | Entry Date: {formatted_data['Entry Date']}</p>
            <p style="margin:0; font-weight:bold; color: {max_gain_color};">Max Gain: {formatted_data['Max Gain']}</p>
            <p style="margin:0; font-weight:bold; color: {rr_ratio_color};">Risk-Reward Ratio: {formatted_data['Risk-Reward Ratio']}</p>
            <p style="margin:0; font-weight:bold; color: {status_color};">Status: {formatted_data['Status']}</p>
            <p style="margin:0; color: #555;">Bid: {formatted_data['Bid']} | Ask: {formatted_data['Ask']}</p>
            <p style="margin:0; color: #555;">Volume: {formatted_data['Volume']} | Open Interest: {formatted_data['Open Interest']}</p>
            <p style="margin:0; color: #555;">Delta: {formatted_data['Delta']} | Gamma: {formatted_data['Gamma']} | Theta: {formatted_data['Theta']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="text-align:center;">
            <p>1st Exit: {formatted_data['1st Exit']}</p>
            <p>2nd Exit: {formatted_data['2nd Exit']}</p>
            <p>3rd Exit: {formatted_data['3rd Exit']}</p>
        </div>
        """, unsafe_allow_html=True)

# Crear gráfico de opciones
st.subheader("Strike vs Open Interest vs Volume")
graph_data = pd.DataFrame({
    "Strike Price": [opt["strike"] for opt in best_contracts],
    "Open Interest": [opt["open_interest"] for opt in best_contracts],
    "Volume": [opt["volume"] for opt in best_contracts]
})
fig = px.scatter(graph_data, x="Strike Price", y="Open Interest", size="Volume",
                 title="Strike vs Open Interest vs Volume", color="Volume")
st.plotly_chart(fig, use_container_width=True)





# Datos procesados para el gráfico de Gamma Exposure
st.subheader("Gamma Exposure Chart")
processed_data = {}
for opt in options_data:
    strike = opt["strike"]
    option_type = opt["option_type"].upper()
    gamma = opt.get("greeks", {}).get("gamma", 0)
    open_interest = opt.get("open_interest", 0)

    if strike not in processed_data:
        processed_data[strike] = {"CALL": {"Gamma": 0, "OI": 0}, "PUT": {"Gamma": 0, "OI": 0}}

    processed_data[strike][option_type]["Gamma"] = gamma
    processed_data[strike][option_type]["OI"] = open_interest

# Generar el gráfico y mostrarlo
gamma_fig = gamma_exposure_chart(processed_data, current_price)
st.plotly_chart(gamma_fig, use_container_width=True)








# Selecciona automáticamente el primer contrato como ejemplo
selected_contract = best_contracts[0]  # Seleccionar el mejor contrato basado en la puntuación
strike_price = selected_contract["strike"]
premium_paid = selected_contract["last"] or selected_contract["bid"]  # Último precio o precio de compra
contract_type = selected_contract["option_type"]
current_price = get_current_price(ticker)

# Verificar que los datos necesarios están disponibles
if premium_paid > 0:
    # Generar gráfico de riesgo/retorno
    st.subheader("Risk/Return Chart")
    risk_return_fig = risk_return_chart_auto(strike_price, premium_paid, current_price, contract_type)
    st.plotly_chart(risk_return_fig, use_container_width=True)
else:
    st.warning("No valid premium price available for the selected contract.")







# --- Despliegue en la app ---
# Calcular volatilidad histórica simulada (sustituir por datos reales si están disponibles)
historical_volatility = 25  # Supongamos 30% como ejemplo

# Generar recomendaciones basadas en IV, HV y tipo de contrato
trade_recommendations = recommend_trades_based_on_iv_hv(options_data, historical_volatility)

# Validar si hay datos antes de mostrar la tabla y el gráfico
if not trade_recommendations:
    st.warning("No hay contratos recomendados basados en IV vs HV.")
else:
    # Crear dataframe de recomendaciones
    recommendations_df = pd.DataFrame(trade_recommendations)

    # Mostrar resultados originales en una tabla
    st.write("### Recomendaciones de Compra/Venta basadas en IV vs HV")
    st.dataframe(recommendations_df)

    # --- Agrupar por Strike y Tipo para simplificar el gráfico ---
    recommendations_df_grouped = (
        recommendations_df.groupby(["Strike", "Type"], as_index=False)
        .agg({
            "IV - HV": "mean",  # Promediar IV - HV por Strike y Tipo
            "Recommendation": lambda x: x.mode()[0] if not x.empty else "Avoid",  # Recomendación dominante
            "Delta": "mean"  # Promediar Delta
        })
    )

    # Crear gráfico de barras con agrupación
    st.write("### Gráfico de Contratos Recomendados (Agrupados por Strike y Tipo)")
    fig_recommendations = px.bar(
        recommendations_df_grouped,
        x="Strike",
        y="IV - HV",
        color="Recommendation",
        facet_col="Type",  # Separar por Calls y Puts
        title="IV - HV, Delta y Tipo",
        labels={"IV - HV": "Diferencia IV - HV", "Type": "Tipo de Contrato"},
        text="Recommendation",
        color_discrete_map={
            "Buy Call": "green",
            "Sell Call": "red",
            "Avoid Call": "gray",
            "Buy Put": "blue",
            "Sell Put": "purple",
            "Avoid Put": "orange"
        }
    )
    st.plotly_chart(fig_recommendations, use_container_width=True)
