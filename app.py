
import streamlit as st  # Para la interfaz Streamlit
import pandas as pd  # Para manipulación de datos tabulares
import requests  # Para llamadas a la API Tradier
import plotly.express as px  # Para gráficos interactivos sencillos
import plotly.graph_objects as go  # Para gráficos avanzados
from datetime import datetime, timedelta  # Para manejo de fechas
import numpy as np  # Para cálculos matemáticos y manipulación de arrays



# Configuración inicial de la página
st.set_page_config(page_title="SCANNER ", layout="wide")

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
        title="(Calls vs Puts)",
        xaxis_title="Strike Price",
        yaxis_title="VOLUME E",
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
        title="GEAR",
        xaxis_title="GEAR",
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
max_pain = calculate_max_pain_optimized(options_data)

# Procesar datos para gráficos con validaciones
processed_data = {}

for opt in options_data:
    # Verificar si el elemento es válido
    if not opt or not isinstance(opt, dict):
        continue  # Ignorar valores inválidos

    # Validar strike
    strike = opt.get("strike")
    if strike is None:
        continue  # Ignorar si no tiene strike

    # Validar option_type
    option_type = opt.get("option_type", "").upper()
    if option_type not in ["CALL", "PUT"]:
        continue  # Ignorar si el tipo de opción no es válido

    # Obtener valores seguros
    gamma = opt.get("greeks", {}).get("gamma", 0)
    open_interest = opt.get("open_interest", 0)

    # Inicializar estructura si no existe
    if strike not in processed_data:
        processed_data[strike] = {"CALL": {"Gamma": 0, "OI": 0}, "PUT": {"Gamma": 0, "OI": 0}}

    # Actualizar datos
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


def calculate_support_resistance_gamma(processed_data, current_price, price_range=21):
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








def calculate_iron_condor(processed_data, current_price, step=5):
    """
    Calcula los strikes, primas, puntos de equilibrio y rango de beneficio para un Iron Condor.
    """
    # Identificar el Gamma más alto para CALLs y PUTs
    max_gamma_call = max(
        (processed_data[strike]["CALL"]["Gamma"], strike)
        for strike in processed_data if "CALL" in processed_data[strike]
    )
    max_gamma_put = max(
        (processed_data[strike]["PUT"]["Gamma"], strike)
        for strike in processed_data if "PUT" in processed_data[strike]
    )

    # Strikes óptimos para vender
    strike_call_sell = max_gamma_call[1]
    strike_put_sell = max_gamma_put[1]

    # Strikes para las posiciones compradas
    strike_call_buy = strike_call_sell + step
    strike_put_buy = strike_put_sell - step

    # Primas para cada posición (usando Gamma * OI como aproximación)
    premium_call_sell = processed_data[strike_call_sell]["CALL"]["OI"] * processed_data[strike_call_sell]["CALL"]["Gamma"]
    premium_call_buy = processed_data.get(strike_call_buy, {}).get("CALL", {}).get("OI", 0) * processed_data.get(strike_call_buy, {}).get("CALL", {}).get("Gamma", 0)

    premium_put_sell = processed_data[strike_put_sell]["PUT"]["OI"] * processed_data[strike_put_sell]["PUT"]["Gamma"]
    premium_put_buy = processed_data.get(strike_put_buy, {}).get("PUT", {}).get("OI", 0) * processed_data.get(strike_put_buy, {}).get("Gamma", 0)

    # Cálculo de los puntos de equilibrio
    breakeven_call = strike_call_sell + (premium_call_sell - premium_call_buy)
    breakeven_put = strike_put_sell - (premium_put_sell - premium_put_buy)

    # Rango de beneficio
    max_profit_range = (breakeven_put, breakeven_call)

    return {
        "Sell Call Strike": strike_call_sell,
        "Buy Call Strike": strike_call_buy,
        "Sell Put Strike": strike_put_sell,
        "Buy Put Strike": strike_put_buy,
        "Breakeven Call": breakeven_call,
        "Breakeven Put": breakeven_put,
        "Max Profit Range": max_profit_range,
        "Premiums": {
            "Call Sell": premium_call_sell,
            "Call Buy": premium_call_buy,
            "Put Sell": premium_put_sell,
            "Put Buy": premium_put_buy,
        }
    }


# Calcular el Iron Condor
iron_condor = calculate_iron_condor(processed_data, current_price)

# Presentar resultados en tarjetas dinámicas
st.subheader("Analysis")

# Color personalizado
text_color = "black"  # Cambia a "black", "yellow", o cualquier color CSS válido

# Tarjeta para las posiciones CALL
st.markdown(f"""
    <div style="border: 2px solid #007bff; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #e6f7ff;">
        <h3 style="color: #0056b3;">CALLs</h3>
        <p style="color: {text_color};"><b>Sell Call Strike:</b> {iron_condor['Sell Call Strike']}</p>
        <p style="color: {text_color};"><b>Buy Call Strike:</b> {iron_condor['Buy Call Strike']}</p>
        <p style="color: {text_color};"><b>Breakeven Call:</b> {iron_condor['Breakeven Call']:.2f}</p>
        <p style="color: {text_color};"><b>Premium (Sell Call):</b> ${iron_condor['Premiums']['Call Sell']:.2f}</p>
        <p style="color: {text_color};"><b>Premium (Buy Call):</b> ${iron_condor['Premiums']['Call Buy']:.2f}</p>
    </div>
""", unsafe_allow_html=True)

# Tarjeta para las posiciones PUT
st.markdown(f"""
    <div style="border: 2px solid #dc3545; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #f8d7da;">
        <h3 style="color: #8b0000;">PUTs</h3>
        <p style="color: {text_color};"><b>Sell Put Strike:</b> {iron_condor['Sell Put Strike']}</p>
        <p style="color: {text_color};"><b>Buy Put Strike:</b> {iron_condor['Buy Put Strike']}</p>
        <p style="color: {text_color};"><b>Breakeven Put:</b> {iron_condor['Breakeven Put']:.2f}</p>
        <p style="color: {text_color};"><b>Premium (Sell Put):</b> ${iron_condor['Premiums']['Put Sell']:.2f}</p>
        <p style="color: {text_color};"><b>Premium (Buy Put):</b> ${iron_condor['Premiums']['Put Buy']:.2f}</p>
    </div>
""", unsafe_allow_html=True)

# Tarjeta para el rango de beneficio máximo
st.markdown(f"""
    <div style="border: 2px solid #28a745; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #d4edda;">
        <h3 style="color: #155724;">Range</h3>
        <p style="color: {text_color};"><b>Desde:</b> {iron_condor['Max Profit Range'][0]:.2f}</p>
        <p style="color: {text_color};"><b>Hasta:</b> {iron_condor['Max Profit Range'][1]:.2f}</p>
    </div>
""", unsafe_allow_html=True)


################################################################################





















# Obtener el precio actual
current_price = get_current_price(ticker)

# Mostrar el precio actual en la interfaz para debugging
st.write(f"**Current Price:** ${current_price:.2f}")

# Verificar si el precio es válido
if current_price == 0:
    st.error("Error: Could not retrieve current price. Please check your API or the ticker symbol.")
    st.stop()

















def detect_synthetic_trigger(processed_data, current_price, threshold_percentage=10):
    """
    Detecta un "sintético" cuando el precio rompe un alto volumen de Gamma PUT o CALL.
    Asegura que el tipo de contrato esté correctamente asociado.
    """
    # Identificar el strike con el mayor volumen de Gamma para CALL y PUT
    max_gamma_call = max(
        (processed_data[strike]["CALL"]["Gamma"] * processed_data[strike]["CALL"]["OI"], strike)
        for strike in processed_data if "CALL" in processed_data[strike]
    )
    max_gamma_put = max(
        (processed_data[strike]["PUT"]["Gamma"] * processed_data[strike]["PUT"]["OI"], strike)
        for strike in processed_data if "PUT" in processed_data[strike]
    )

    # Definir los niveles de ruptura
    gamma_call_level = max_gamma_call[1]
    gamma_put_level = max_gamma_put[1]
    call_threshold_up = gamma_call_level * (1 + threshold_percentage / 100)
    put_threshold_up = gamma_put_level * (1 + threshold_percentage / 100)

    # Inicializar los triggers
    synthetic_trigger = None

    # Verificar ruptura en CALLs
    if current_price > call_threshold_up:  # Ruptura hacia arriba en CALLs
        next_targets = sorted(
            [(processed_data[strike]["CALL"]["Gamma"] * processed_data[strike]["CALL"]["OI"], strike)
             for strike in processed_data if strike > gamma_call_level and "CALL" in processed_data[strike]],
            reverse=True
        )[:3]
        next_targets = [strike for _, strike in next_targets]
        synthetic_trigger = {
            "type": "CALL",
            "triggered_level": gamma_call_level,
            "next_targets": next_targets,
            "direction": "UP"
        }

    # Verificar ruptura en PUTs
    elif current_price > put_threshold_up:  # Ruptura hacia arriba en PUTs
        next_targets = sorted(
            [(processed_data[strike]["PUT"]["Gamma"] * processed_data[strike]["PUT"]["OI"], strike)
             for strike in processed_data if strike > gamma_put_level and "PUT" in processed_data[strike]],
            reverse=True
        )[:3]
        next_targets = [strike for _, strike in next_targets]
        synthetic_trigger = {
            "type": "PUT",
            "triggered_level": gamma_put_level,
            "next_targets": next_targets,
            "direction": "UP"
        }

    # Si no hay ruptura, mostrar los siguientes targets relevantes
    if synthetic_trigger is None:
        next_call_target = min(
            strike for strike in processed_data if strike > current_price and "CALL" in processed_data[strike]
        )
        next_put_target = max(
            strike for strike in processed_data if strike < current_price and "PUT" in processed_data[strike]
        )
        synthetic_trigger = {
            "type": "NO TRIGGER",
            "next_call_target": next_call_target,
            "next_put_target": next_put_target,
        }

    return synthetic_trigger

# Detectar el sintético
synthetic_trigger = detect_synthetic_trigger(processed_data, current_price, threshold_percentage=11)

# Mostrar resultados
if synthetic_trigger["type"] == "NO TRIGGER":
    st.markdown(f"""
        <div style="border: 2px solid #ffc107; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #fff3cd;">
            <h3 style="color: black;">Direction</h3>
            <p style="color: black;">Current price: ${current_price:.2f}</p>
            <p style="color: black;"><b>Next CALL Target:</b> {synthetic_trigger['next_call_target']}</p>
            <p style="color: black;"><b>Next PUT Target:</b> {synthetic_trigger['next_put_target']}</p>
        </div>
    """, unsafe_allow_html=True)
else:
    next_targets = ", ".join(map(str, synthetic_trigger['next_targets']))
    st.markdown(f"""
        <div style="border: 2px solid #ffc107; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #fff3cd;">
            <h3 style="color: black;">Synthetic Trigger Detected</h3>
            <p style="color: black;"><b>Type:</b> {synthetic_trigger['type']}</p>
            <p style="color: black;"><b>Triggered Level:</b> {synthetic_trigger['triggered_level']}</p>
            <p style="color: black;"><b>Direction:</b> {synthetic_trigger['direction']}</p>
            <p style="color: black;"><b>Next Targets:</b> {next_targets}</p>
        </div>
    """, unsafe_allow_html=True)





















def detect_and_update_targets(processed_data, current_price, threshold_percentage=10):
    """
    Detect and dynamically update Maximum and Bottom Targets if they are broken.
    """
    # Identificar los strikes con el Gamma más alto
    max_gamma_call = max(
        (processed_data[strike]["CALL"]["Gamma"] * processed_data[strike]["CALL"]["OI"], strike)
        for strike in processed_data if "CALL" in processed_data[strike]
    )
    max_gamma_put = max(
        (processed_data[strike]["PUT"]["Gamma"] * processed_data[strike]["PUT"]["OI"], strike)
        for strike in processed_data if "PUT" in processed_data[strike]
    )

    # Extraer strikes iniciales
    strike_call = max_gamma_call[1]
    strike_put = max_gamma_put[1]

    # Si rompe el máximo
    if current_price > strike_call * (1 + threshold_percentage / 100):
        # Buscar el siguiente nivel CALL
        next_call_targets = sorted(
            [(processed_data[s]["CALL"]["Gamma"] * processed_data[s]["CALL"]["OI"], s)
             for s in processed_data if s > strike_call and "CALL" in processed_data[s]],
            reverse=True
        )
        strike_call = next_call_targets[0][1] if next_call_targets else strike_call  # Actualizar si hay otro nivel

    # Si rompe el mínimo
    if current_price < strike_put * (1 - threshold_percentage / 100):
        # Buscar el siguiente nivel PUT
        next_put_targets = sorted(
            [(processed_data[s]["PUT"]["Gamma"] * processed_data[s]["PUT"]["OI"], s)
             for s in processed_data if s < strike_put and "PUT" in processed_data[s]],
            reverse=True
        )
        strike_put = next_put_targets[0][1] if next_put_targets else strike_put  # Actualizar si hay otro nivel

    return {
        "Current Price": current_price,
        "Maximum Target Today": strike_call,
        "Bottom Target Today": strike_put,
        "Next CALL Target": round(strike_call, 2),
        "Next PUT Target": round(strike_put, 2),
    }


# Calcular y actualizar dinámicamente los targets
updated_targets = detect_and_update_targets(processed_data, current_price)

# Mostrar resultados
st.subheader("Target Updates")
st.markdown(f"""
    <div style="border: 2px solid #007bff; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #e6f7ff;">
        <h3 style="color: #0056b3;">TODAY TARGETS</h3>
        <p style="color: black;"><b>Current Price:</b> ${updated_targets['Current Price']:.2f}</p>
        <p style="color: black;"><b>Maximum Target Today:</b> ${updated_targets['Maximum Target Today']}</p>
        <p style="color: black;"><b>Bottom Target Today:</b> ${updated_targets['Bottom Target Today']}</p>
    </div>
""", unsafe_allow_html=True)

# Mostrar los siguientes targets dinámicos
st.markdown(f"""
    <div style="border: 2px solid #28a745; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: #d4edda;">
        <h3 style="color: #155724;">Break the Targets</h3>
        <p style="color: black;"><b>Next CALL Target:</b> ${updated_targets['Next CALL Target']}</p>
        <p style="color: black;"><b>Next PUT Target:</b> ${updated_targets['Next PUT Target']}</p>
    </div>
""", unsafe_allow_html=True)



















#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>










# Función para calcular zonas clave
def calculate_key_levels(options_data):
    key_levels = {"CALL": [], "PUT": []}
    for option in options_data:
        strike = option["strike"]
        gamma = option.get("greeks", {}).get("gamma", 0)
        delta = option.get("greeks", {}).get("delta", 0)
        oi = option.get("open_interest", 0)

        if option["option_type"].upper() == "CALL":
            key_levels["CALL"].append((strike, gamma, delta, oi))
        elif option["option_type"].upper() == "PUT":
            key_levels["PUT"].append((strike, gamma, delta, oi))

    # Ordenar por Open Interest y seleccionar los 3 niveles más relevantes
    for key in key_levels:
        key_levels[key] = sorted(key_levels[key], key=lambda x: x[3], reverse=True)[:3]

    return key_levels

# Función para verificar alertas dinámicas
def check_alerts(current_price, key_levels):
    alerts = []
    for option_type, levels in key_levels.items():
        for strike, gamma, delta, oi in levels:
            if abs(current_price - strike) <= 1:  # A menos de 1 punto del strike
                alerts.append(f"{option_type} Strike {strike} is being approached! (Gamma: {gamma}, OI: {oi})")
    return alerts

# Interfaz de usuario



current_price = get_current_price(ticker)
options_data = get_options_data(ticker, expiration_date)
import time
time.sleep(0.2)  # Pausa de 200ms entre solicitudes


if not options_data:
    st.error("No options data available.")
    st.stop()

# Calcular niveles clave y alertas
key_levels = calculate_key_levels(options_data)
alerts = check_alerts(current_price, key_levels)

# Mostrar resultados
st.subheader("Current Price")
st.markdown(f"**${current_price:.2f}**")

st.subheader("Key Levels")
for option_type, levels in key_levels.items():
    st.markdown(f"### {option_type}")
    for strike, gamma, delta, oi in levels:
        st.markdown(f"- **Strike:** {strike}, **Gamma:** {gamma:.4f}, **Delta:** {delta:.4f}, **OI:** {oi}")

st.subheader("Intraday ")
if alerts:
    for alert in alerts:
        st.markdown(f"🚨 {alert}")
else:
    st.markdown("No alerts at the moment.")

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

st.plotly_chart(plot_gamma_oi(key_levels), use_container_width=True)









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

# Función para calcular top movers básicos
def calculate_top_movers(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    df["IV"] = df["IV"].fillna(0).infer_objects(copy=False)
    df["Average Volume"] = df["Average Volume"].replace(0, np.nan)
    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Cambio Relativo"] = np.abs(df["Change (%)"]) / df["Change (%)"].mean()

    df["Score"] = (df["Volumen Relativo"] * 4) + (df["Cambio Relativo"] * 3) + df["IV"]

    return df.sort_values("Score", ascending=False).head(3)

# Función para detectar movimientos continuos
def calculate_continuous_movers(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    df["IV"] = df["IV"].fillna(0).infer_objects(copy=False)
    df["HV"] = df["HV"].fillna(0).infer_objects(copy=False)
    df["Average Volume"] = df["Average Volume"].replace(0, np.nan)

    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Momentum"] = np.abs(df["Price"] - df["Previous Close"]) / df["Previous Close"]
    df["Cambio Relativo"] = np.abs(df["Change (%)"]) / df["Change (%)"].mean()

    df["Score"] = (df["Volumen Relativo"] * 4) + (df["Momentum"] * 3) + (df["Cambio Relativo"] * 2) + (df["IV"] + df["HV"])

    return df.sort_values("Score", ascending=False).head(3)

# Función para calcular potencial explosivo
def calculate_explosive_movers(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    df["IV"] = df["IV"].fillna(0).infer_objects(copy=False)
    df["HV"] = df["HV"].fillna(0).infer_objects(copy=False)
    df["Average Volume"] = df["Average Volume"].replace(0, np.nan)

    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Explosión"] = df["Volumen Relativo"] * df["Change (%)"].abs()
    df["Score"] = df["Explosión"] + (df["IV"] * 0.5)

    return df.sort_values("Score", ascending=False).head(3)




# Función para calcular actividad de opciones
def calculate_options_activity(data):
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    df["IV"] = df["IV"].fillna(0).astype(float)
    df["Volumen Relativo"] = df["Volume"] / df["Average Volume"]
    df["Options Activity"] = df["Volumen Relativo"] * df["IV"]

    return df.sort_values("Options Activity", ascending=False).head(3)


# Interfaz de usuario
st.title("")



stock_data = fetch_batch_stock_data(all_tickers)

if stock_data:
    # Versión 1: Básico
    st.subheader("Top Movers")
    top_movers = calculate_top_movers(stock_data)
    for _, row in top_movers.iterrows():
        st.markdown(f"""
            <div style="border: 2px solid #28a745; padding: 10px; margin-bottom: 10px;">
                <h3>{row['Ticker']}</h3>
                <p><b>Precio:</b> ${row['Price']:.2f}</p>
                <p><b>Cambio (%):</b> {row['Change (%)']:.2f}%</p>
                <p><b>Volumen Relativo:</b> {row['Volumen Relativo']:.2f}</p>
                <p><b>Puntaje:</b> {row['Score']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    # Versión 2: Movimientos Continuos
    st.subheader("")
    continuous_movers = calculate_continuous_movers(stock_data)
    for _, row in continuous_movers.iterrows():
        st.markdown(f"""
            <div style="border: 2px solid #ff4500; padding: 10px; margin-bottom: 10px;">
                <h3>{row['Ticker']}</h3>
                <p><b>Precio:</b> ${row['Price']:.2f}</p>
                <p><b>Momentum:</b> {row['Momentum']:.2f}</p>
                <p><b>Puntaje:</b> {row['Score']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)

    # Versión 3: Explosión
    st.subheader("")
    explosive_movers = calculate_explosive_movers(stock_data)
    for _, row in explosive_movers.iterrows():
        st.markdown(f"""
            <div style="border: 2px solid #0000FF; padding: 10px; margin-bottom: 10px;">
                <h3>{row['Ticker']}</h3>
                <p><b>Precio:</b> ${row['Price']:.2f}</p>
                <p><b>Explosión:</b> {row['Explosión']:.2f}</p>
                <p><b>Puntaje:</b> {row['Score']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.warning("No se encontraron datos para los tickers ingresados.")






# Nueva Sección: Actividad de Opciones
    st.subheader("Actividad de Opciones")
    options_activity = calculate_options_activity(stock_data)
    for _, row in options_activity.iterrows():
        st.markdown(f"""
            <div style="border: 2px solid #0000FF; padding: 10px; margin-bottom: 10px;">
                <h3>{row['Ticker']}</h3>
                <p><b>Precio:</b> ${row['Price']:.2f}</p>
                <p><b>Volumen Relativo:</b> {row['Volumen Relativo']:.2f}</p>
                <p><b>Actividad de Opciones:</b> {row['Options Activity']:.2f}</p>
            </div>
        """, unsafe_allow_html=True)


    else:
     st.warning("No se encontraron datos para los tickers ingresados.")






import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

# Función para buscar las últimas noticias
def fetch_latest_news(keywords):
    base_url = "https://www.google.com/search"
    query = "+".join(keywords)
    params = {"q": query, "tbm": "nws", "tbs": "qdr:m"}  # Último minuto

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    news = []

    # Intentar múltiples selectores para compatibilidad
    articles = soup.select("div.dbsr") or soup.select("div.Gx5Zad.fP1Qef.xpd.EtOod.pkphOe")
    
    for article in articles[:20]:  # Limitar a las 10 noticias más recientes
        title_tag = article.select_one("div.JheGif.nDgy9d") or article.select_one("div.BNeawe.vvjwJb.AP7Wnd")
        link_tag = article.a
        if title_tag and link_tag:
            title = title_tag.text.strip()
            link = link_tag["href"]
            time_tag = article.select_one("span.WG9SHc")
            time_posted = time_tag.text if time_tag else "Just now"
            news.append({"title": title, "link": link, "time": time_posted})
    
    return news

# Función de respaldo con Bing News
def fetch_backup_news(keywords):
    base_url = "https://www.bing.com/news/search"
    query = " ".join(keywords)
    params = {"q": query, "qft": "+filterui:age-lt60"}  # Últimos 60 minutos

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    news = []

    articles = soup.select("a.title")
    for article in articles[:200]:
        title = article.text.strip()
        link = article["href"]
        news.append({"title": title, "link": link, "time": "Recently"})
    
    return news

# Configuración de Streamlit
st.title("Live News Scanner")
st.write("")

# Palabras clave automáticas
keywords = ["CPI", "Core CPI", "PCE", "Core PCE", "Inflation", "Elon",
    "GDP", "Trump", "Donald Trump",
    "Unemployment", "Nonfarm Payrolls", "Jobless Claims", "Labor Market", "Employment Rate",
    "FOMC", "Fed Minutes", "Interest Rate Decision", "", "Tapering",
    "Rate Hike", "Rate Cut", "Treasury Yield", "Debt Ceiling", "Treasury Bonds", "Yield Curve",
    "Consumer Sentiment", "ISM Manufacturing","upgrade", "stock buyback", 
    "price target", "breaking news", "Apple", "Microsoft", "Google", "Amazon", 
    "Meta", "Tesla", "NVDA"]

# Inicializar estado
if "news_data" not in st.session_state:
    st.session_state.news_data = []

# Contenedor dinámico para mostrar las noticias
news_placeholder = st.empty()

# Loop para actualizar automáticamente cada 30 segundos
while True:
    # Obtener noticias desde Google
    with st.spinner("Fetching breaking news..."):
        latest_news = fetch_latest_news(keywords)
    
    # Si Google no devuelve resultados, usar Bing
    if not latest_news:
        latest_news = fetch_backup_news(keywords)

    # Actualizar el contenedor con las noticias
    if latest_news:
        st.session_state.news_data = latest_news
        with news_placeholder.container():
            st.success(f"Latest news updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!")
            for idx, article in enumerate(latest_news, 1):
                st.markdown(f"### {idx}. [{article['title']}]({article['link']})")
                st.markdown(f"**Published:** {article['time']}")
    else:
        st.error("No recent news found from any source.")

    # Esperar 30 segundos antes de actualizar
    time.sleep(80)

