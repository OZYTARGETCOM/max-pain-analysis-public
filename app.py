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
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import openai
from dotenv import load_dotenv


# Configuración inicial de la página
st.set_page_config(page_title="SCANNER OPTIONS", layout="wide")

# Configuración de la API Tradier
API_KEY = "wMG8GrrZMBFeZMCWJTqTzZns7B4w"
BASE_URL = "https://api.tradier.com/v1"

# Función para obtener datos de opciones
@st.cache_data
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
def gamma_exposure_chart_optimized(processed_data, current_price, max_pain):
    strikes = sorted(processed_data.keys())

    gamma_calls = [processed_data[s]["CALL"]["OI"] * processed_data[s]["CALL"]["Gamma"] for s in strikes]
    gamma_puts = [-processed_data[s]["PUT"]["OI"] * processed_data[s]["PUT"]["Gamma"] for s in strikes]

    fig = go.Figure()

    # Añadir barras de Gamma Calls y Gamma Puts
    fig.add_trace(go.Bar(x=strikes, y=gamma_calls, name="Gamma CALL", marker_color="blue"))
    fig.add_trace(go.Bar(x=strikes, y=gamma_puts, name="Gamma PUT", marker_color="red"))

    # Línea de Precio Actual
    fig.add_shape(
        type="line",
        x0=current_price, x1=current_price,
        y0=min(gamma_calls + gamma_puts) * 1.1,
        y1=max(gamma_calls + gamma_puts) * 1.1,
        line=dict(color="orange", dash="dot", width=1),  # Línea más delgada
    )

    # Línea de Max Pain
    fig.add_shape(
        type="line",
        x0=max_pain, x1=max_pain,
        y0=min(gamma_calls + gamma_puts) * 1.1,
        y1=max(gamma_calls + gamma_puts) * 1.1,
        line=dict(color="green", dash="dash", width=1),  # Línea más delgada
    )

    # Etiqueta para el Precio Actual
    fig.add_annotation(
        x=current_price,
        y=max(gamma_calls + gamma_puts) * 0.9,
        text=f"Current Price: ${current_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="orange",
        font=dict(color="orange", size=12),
    )

    # Etiqueta para Max Pain
    fig.add_annotation(
        x=max_pain,
        y=max(gamma_calls + gamma_puts) * 0.8,
        text=f"Max Pain: ${max_pain:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green",
        font=dict(color="green", size=12),
    )

    fig.update_layout(
        title="Gamma Exposure (Calls vs Puts)",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure",
        template="plotly_white",
        legend=dict(title="Option Type"),
    )
    return fig


# Función para crear Heatmap
def create_heatmap(processed_data):
    strikes = sorted(processed_data.keys())

    oi = [processed_data[s]["CALL"]["OI"] + processed_data[s]["PUT"]["OI"] for s in strikes]
    gamma = [processed_data[s]["CALL"]["Gamma"] + processed_data[s]["PUT"]["Gamma"] for s in strikes]
    volume = [processed_data[s]["CALL"]["OI"] * processed_data[s]["CALL"]["Gamma"] +
              processed_data[s]["PUT"]["OI"] * processed_data[s]["PUT"]["Gamma"] for s in strikes]

    data = pd.DataFrame({
        'OI': oi,
        'Gamma': gamma,
        'Volume': volume
    })

    data_normalized = data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    fig = go.Figure(data=go.Heatmap(
        z=data_normalized.T.values,
        x=strikes,
        y=data_normalized.columns,
        colorscale='Viridis',
        colorbar=dict(title='Normalized Value'),
    ))

    fig.update_layout(
        title="Supports & Resistances (Heatmap)",
        xaxis_title="Strike Price",
        yaxis_title="Metrics",
        template="plotly_dark"
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
st.title("SCANNER OPTIONS")

ticker = st.text_input("Ticker", value="AAPL", key="ticker_input").upper()
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

# Calcular Max Pain con el cálculo mejorado
max_pain = calculate_max_pain_optimized(options_data)

# Procesar datos para gráficos
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

# Mostrar gráficos
st.subheader("Gamma Exposure")
gamma_fig = gamma_exposure_chart_optimized(processed_data, current_price, max_pain)
st.plotly_chart(gamma_fig, use_container_width=True)
st.write(f"**Max Pain Calculated:** ${max_pain}")
st.write(f"**Current Price:** ${current_price:.2f}")



st.subheader("Heatmap")
heatmap_fig = create_heatmap(processed_data)
st.plotly_chart(heatmap_fig, use_container_width=True)






st.subheader("Skew Analysis")

# Llamar a la función mejorada
skew_fig, total_calls, total_puts = plot_skew_analysis_with_totals(options_data)

# Mostrar los totales en Streamlit
st.write(f"**Total CALLS** {total_calls}")
st.write(f"**Total PUTS** {total_puts}")

# Mostrar el gráfico
st.plotly_chart(skew_fig, use_container_width=True)



# Función para generar señales en formato de tarjetas






























































#########################################################################


def calculate_support_resistance_gamma(processed_data, current_price, price_range=20):
    """
    Calcula el soporte y la resistencia basados en el Gamma más alto dentro de un rango dado.
    """
    max_gamma_call, max_gamma_put = -1, -1
    resistance_strike, support_strike = None, None

    # Filtrar strikes en el rango deseado
    strikes_in_range = {
        strike: data for strike, data in processed_data.items()
        if current_price - price_range <= strike <= current_price + price_range
    }

    # Buscar la resistencia (CALL con mayor Gamma)
    for strike, data in strikes_in_range.items():
        gamma_call = data["CALL"].get("Gamma", 0)
        if strike >= current_price and gamma_call > max_gamma_call:
            max_gamma_call = gamma_call
            resistance_strike = strike

    # Buscar el soporte (PUT con mayor Gamma)
    for strike, data in strikes_in_range.items():
        gamma_put = data["PUT"].get("Gamma", 0)
        if strike <= current_price and gamma_put > max_gamma_put:
            max_gamma_put = gamma_put
            support_strike = strike

    # Validar resultados
    return {
        "Resistance (CALL)": {
            "Strike": resistance_strike or "No Match",
            "Gamma": max_gamma_call if resistance_strike else 0,
        },
        "Support (PUT)": {
            "Strike": support_strike or "No Match",
            "Gamma": max_gamma_put if support_strike else 0,
        },
    }


def generate_winning_contract(options_data, current_price, iv_hv_ratio=1.2):
    """
    Genera un contrato ganador basado en criterios calculados.
    """
    winning_contracts = []

    for option in options_data:
        strike = option["strike"]
        delta = option.get("greeks", {}).get("delta", 0)
        gamma = option.get("greeks", {}).get("gamma", 0)
        theta = option.get("greeks", {}).get("theta", 0)
        iv = option.get("implied_volatility", 0) * 100
        hv = option.get("historical_volatility", 0) * 100
        volume = option.get("volume", 0)
        open_interest = option.get("open_interest", 0)
        bid = option["bid"] or 0
        ask = option["ask"] or 0
        mid_price = (bid + ask) / 2 if bid and ask else bid or ask

        # Condiciones del contrato ganador
        if (
            0.4 <= abs(delta) <= 0.6 and
            gamma > 0.01 and
            mid_price > 0 and
            volume > 500 and
            open_interest > 1000 and
            (iv / hv if hv > 0 else 1) <= iv_hv_ratio
        ):
            max_gain = ((current_price - strike) / mid_price * 100) if delta < 0 else ((strike - current_price) / mid_price * 100)
            risk_reward = max_gain / mid_price if mid_price > 0 else 0

            winning_contracts.append({
                "Strike": strike,
                "Type": "CALL" if delta > 0 else "PUT",
                "Delta": round(delta, 4),
                "Gamma": round(gamma, 4),
                "Theta": round(theta, 4),
                "IV": round(iv, 2),
                "HV": round(hv, 2),
                "Volume": volume,
                "Open Interest": open_interest,
                "Max Gain (%)": round(max_gain, 2),
                "Risk-Reward Ratio": round(risk_reward, 2),
                "Entry Price": round(mid_price, 2),
            })

    # Ordenar por el mayor Max Gain
    winning_contracts = sorted(winning_contracts, key=lambda x: x["Max Gain (%)"], reverse=True)
    return winning_contracts[:3]


def display_support_resistance(support_resistance):
    """
    Muestra el soporte y resistencia calculados en tarjetas dinámicas.
    """
    st.subheader("")
    for key, value in support_resistance.items():
        card_color = "#d4f4dd" if "CALL" in key else "#f4d4d4"
        border_color = "#28a745" if "CALL" in key else "#dc3545"

        st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 10px; margin-bottom: 10px; background-color: {card_color};">
                <h4 style="color: black; margin-bottom: 5px;">{key}</h4>
                <p style="color: black; margin: 5px 0;"><b>Strike:</b> {value['Strike']}</p>
                <p style="color: black; margin: 5px 0;"><b>Gamma:</b> {value['Gamma']:.4f}</p>
            </div>
        """, unsafe_allow_html=True)


def display_winning_contracts(winning_contracts):
    """
    Muestra los contratos ganadores en tarjetas dentro de Streamlit.
    Las tarjetas serán verdes para CALLs y rojas para PUTs.
    """
    if not winning_contracts:
        st.write("Wait Dude There are no Contracts Relax.")
        return

    st.subheader("High Performance Contracts")
    for contract in winning_contracts:
        # Determinar el color de la tarjeta según el tipo de contrato
        card_color = "#d4f4dd" if contract['Type'] == "CALL" else "#f4d4d4"
        border_color = "#28a745" if contract['Type'] == "CALL" else "#dc3545"

        # Contenido dinámico de la tarjeta
        st.markdown(f"""
            <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: {card_color};">
                <h4 style="color: black; margin-bottom: 10px; font-size: 18px;">{contract['Type']} - Strike: {contract['Strike']}</h4>
                <p style="color: black; margin: 5px 0;"><b>Delta:</b> {contract['Delta']} | <b>Gamma:</b> {contract['Gamma']} | <b>Theta:</b> {contract['Theta']}</p>
                <p style="color: black; margin: 5px 0;"><b>IV:</b> {contract['IV']}% | <b>HV:</b> {contract['HV']}%</p>
                <p style="color: black; margin: 5px 0;"><b>Volume:</b> {contract['Volume']} | <b>Open Interest:</b> {contract['Open Interest']}</p>
                <p style="color: black; margin: 5px 0;"><b>Entry Price:</b> ${contract['Entry Price']} | <b>Max Gain:</b> {contract['Max Gain (%)']}% | <b>RR Ratio:</b> {contract['Risk-Reward Ratio']}</p>
            </div>
        """, unsafe_allow_html=True)


# Llamar a las funciones y mostrar resultados en Streamlit
support_resistance = calculate_support_resistance_gamma(processed_data, current_price, price_range=20)
display_support_resistance(support_resistance)

winning_options = generate_winning_contract(options_data, current_price)
display_winning_contracts(winning_options)










#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>








