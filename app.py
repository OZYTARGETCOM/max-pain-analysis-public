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
import requests
from datetime import datetime, timedelta

# Configuración inicial de la página
st.set_page_config(page_title="SCANNER OPTIONS", layout="wide")

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>seguridad



# Configuración de la API Tradier
API_KEY = "wMG8GrrZMBFeZMCWJTqTzZns7B4w"
BASE_URL = "https://api.tradier.com/v1"
# Configuración de la API de Noticias
NEWS_API_KEY = "dc681719f9854b148abf6fc1c94fdb33"  # API KEY para NewsAPI
NEWS_BASE_URL = "https://newsapi.org/v2/everything"  # Endpoint de NewsAPI



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
    
    # Contenedor dinámico para mostrar el precio actualizado
price_placeholder = st.empty()

# Función para actualizar dinámicamente el precio
def update_current_price(ticker):
    while True:
        price = get_current_price(ticker)
        price_placeholder.write(f"**Current Price for {ticker}:** ${price:.2f}")
        time.sleep(10)  # Actualiza cada 10 segundos







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
    best_contracts = sorted(best_contracts, key=lambda x: x["score"], reverse=True)[:8]
    return best_contracts


import plotly.graph_objects as go

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

# Función para calcular Max Pain (ajustado)
def calculate_adjusted_max_pain(options_data):
    strikes = {}
    for option in options_data:
        strike = option["strike"]
        open_interest = option["open_interest"] or 0
        if strike not in strikes:
            strikes[strike] = 0
        strikes[strike] += open_interest
    return max(strikes, key=strikes.get)

# Contenedor dinámico para mostrar el precio actualizado
price_placeholder = st.empty()

# Función para actualizar dinámicamente el precio
def update_current_price(ticker):
    while True:
        price = get_current_price(ticker)
        price_placeholder.write(f"**Current Price for {ticker}:** ${price:.2f}")
        time.sleep(10)  # Actualiza cada 10 segundos


def gamma_exposure_chart(strikes_data, current_price, max_pain):
    strikes = sorted(strikes_data.keys())

    gamma_calls = [strikes_data[s]["CALL"]["OI"] * strikes_data[s]["CALL"]["Gamma"] for s in strikes]
    gamma_puts = [strikes_data[s]["PUT"]["OI"] * strikes_data[s]["PUT"]["Gamma"] for s in strikes]

    fig = go.Figure()

    # Añadir barras para Gamma Calls y Puts
    fig.add_trace(go.Bar(
        x=strikes,
        y=gamma_calls,
        name="Gamma CALL",
        marker_color="blue"
    ))
    fig.add_trace(go.Bar(
        x=strikes,
        y=[-g for g in gamma_puts],
        name="Gamma PUT",
        marker_color="red"
    ))

    # Línea para el precio actual
    fig.add_shape(
        type="line",
        x0=current_price,
        x1=current_price,
        y0=min(-max(gamma_puts), min(gamma_calls)) * 1.1,
        y1=max(max(gamma_calls), -min(gamma_puts)) * 1.1,
        line=dict(color="orange", width=2, dash="dot"),
        name="Current Price"
    )

    # Línea para Max Pain
    fig.add_shape(
        type="line",
        x0=max_pain,
        x1=max_pain,
        y0=min(-max(gamma_puts), min(gamma_calls)) * 1.1,
        y1=max(max(gamma_calls), -min(gamma_puts)) * 1.1,
        line=dict(color="green", width=2, dash="dash"),
        name="Max Pain"
    )

    # Etiquetas para el precio actual y Max Pain
    fig.add_annotation(
        x=current_price,
        y=max(max(gamma_calls), -min(gamma_puts)) * 0.9,
        text=f"Current Price: ${current_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="orange",
        font=dict(color="orange", size=12)
    )
    fig.add_annotation(
        x=max_pain,
        y=max(max(gamma_calls), -min(gamma_puts)) * 0.8,
        text=f"Max Pain: ${max_pain:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green",
        font=dict(color="green", size=12)
    )

    # Configuración del diseño del gráfico
    fig.update_layout(
        title="Gamma Exposure (Calls vs Puts)",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure",
        barmode="relative",
        template="plotly_white",
        legend=dict(title="Option Type")
    )

    return fig





# Función para actualizar Max Pain y Precio Actual en Tiempo Real
def refresh_data_in_real_time(ticker, expiration_date, update_interval=10):
    # Contenedores dinámicos en Streamlit
    price_placeholder = st.empty()
    max_pain_placeholder = st.empty()

    while True:
        # Obtener el precio actual y los datos de opciones
        current_price = get_current_price(ticker)
        options_data = get_options_data(ticker, expiration_date)
        
        if not options_data:
            st.error("Error: No se pudieron obtener los datos de opciones.")
            break
        
        max_pain = calculate_adjusted_max_pain(options_data)

        # Actualizar los valores en la interfaz
        price_placeholder.markdown(f"### **Precio Actual ({ticker}):** ${current_price:.2f}")
        max_pain_placeholder.markdown(f"### **Max Pain:** ${max_pain:.2f}")

        # Esperar el intervalo especificado antes de actualizar nuevamente
        time.sleep(update_interval)











# Contenedor dinámico para mostrar el precio actualizado
price_placeholder = st.empty()

# Función para actualizar dinámicamente el precio
def update_current_price(ticker):
    while True:
        price = get_current_price(ticker)
        price_placeholder.write(f"**Current Price for {ticker}:** ${price:.2f}")
        time.sleep(10)  # Actualiza cada 10 segundos


# Función para crear el heatmap
def create_heatmap(processed_data):
    strikes = sorted(processed_data.keys())

    oi = [processed_data[s]["CALL"]["OI"] + processed_data[s]["PUT"]["OI"] for s in strikes]
    gamma = [processed_data[s]["CALL"]["Gamma"] + processed_data[s]["PUT"]["Gamma"] for s in strikes]
    volume = [processed_data[s]["CALL"]["OI"] * processed_data[s]["CALL"]["Gamma"] +
              processed_data[s]["PUT"]["OI"] * processed_data[s]["PUT"]["Gamma"] for s in strikes]

    # Normalización de las métricas
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
        title="Supports & Resistences",
        xaxis_title="Strike Price",
        yaxis_title="Métrica",
        template="plotly_dark"
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
        title=f"Risk/Return MM for {contract_type.upper()}",
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





#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>FUNCION DE VERIFICACION DE CONTRATOS  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 







#######################











#>>>>>>>>>>>>>>>>>>>>>>>>NEWS>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>














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
st.title("SCANNER")



ticker = st.text_input("Ticker", value="NVDA").upper()
expiration_dates = get_expiration_dates(ticker)
if expiration_dates:
    expiration_date = st.selectbox("Expiration Date", expiration_dates)
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

graph_data = pd.DataFrame({
    "Strike Price": [opt["strike"] for opt in best_contracts],
    "Open Interest": [opt["open_interest"] for opt in best_contracts],
    "Volume": [opt["volume"] for opt in best_contracts]
})
fig = px.scatter(graph_data, x="Strike Price", y="Open Interest", size="Volume",
                 title="Strike vs Open Interest vs Volume", color="Volume")
st.plotly_chart(fig, use_container_width=True)














# Selecciona automáticamente el primer contrato como ejemplo
selected_contract = best_contracts[0]  # Seleccionar el mejor contrato basado en la puntuación
strike_price = selected_contract["strike"]
premium_paid = selected_contract["last"] or selected_contract["bid"]  # Último precio o precio de compra
contract_type = selected_contract["option_type"]
current_price = get_current_price(ticker)

# Verificar que los datos necesarios están disponibles
if premium_paid > 0:
    # Generar gráfico de riesgo/retorno
    st.subheader("")
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
    st.write("### Recommended Options")
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
    st.write("")
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


# Interfaz de usuario




# Calcular Max Pain ajustado
adjusted_max_pain = calculate_adjusted_max_pain(options_data)

# Procesar los datos de opciones para generar un diccionario de strikes
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

# Generar el gráfico de Gamma Exposure
st.subheader("")
gamma_fig = gamma_exposure_chart(processed_data, current_price, adjusted_max_pain)
st.plotly_chart(gamma_fig, use_container_width=True)

# Crear el Heatmap
st.subheader("")
heatmap_fig = create_heatmap(processed_data)
st.plotly_chart(heatmap_fig, use_container_width=True)












#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>NEWS





