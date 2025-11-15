# EL OBJETIVO DE ESTE PROGRAMA ES EL SIGUIENTE:
# 1.- Descargar los históricos de las principales criptomonedas: Bitcoin, Ethereum, Binance Coin, Ripple, Bitcoin Cash, Ethereum Classic, Doge (desde enero del 2021).
# 2.- Generar un archivo EXCEL con los datos históricos descargados de cada criptomoneda.

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import numpy as np
import json
import pandas_ta as ta

# Función principal para crear el DataFrame con los datos históricos
def crearDataFrame():

    #Configurar el json de las variables
    with open("config_dev.json", "r") as f:
       config = json.load(f)

    # Define los parámetros de la solicitud a la API de Binance
    symbol = 'BTCUSDT'  # Par de criptomonedas a analizar (Bitcoin contra USDT)
    # interval = input("Intervalo de tiempo (ej. 5m, 30m, 1h, 6h, 1d): \n")  # Intervalo de tiempo de las velas (4 horas)
    timeframe = config.get("common", {}).get("timeframe")
    apalancamiento = config.get("common", {}).get("apalancamiento")
    fecha_inicio_str = config.get("common", {}).get("fecha_inicio")
    fecha_fin_str = config.get("common", {}).get("fecha_fin")
    print(f'Timeframe cargado: {timeframe}')
    print(f'Apalancamiento: {apalancamiento}')
    limit = 1500  # Límite de datos por solicitud (Maximo permitido por la API)

    # Establece la zona horaria de Ciudad de México
    tz_mexico = timezone(timedelta(hours=-6))

    # #Solicita al usuario la fecha e inicio en fomato 'YYYY-MM-DD'
    # fecha_inicio_str = input("Ingrese la fecha de inicio (formato YYYY-MM-DD): \n")

    #convierte la fecha ingresada a un objeto datetime
    try:
        fecha_inicio = datetime.strptime(fecha_inicio_str, "%Y-%m-%d") #Convierte a datetime
        fecha_inicio = fecha_inicio.replace(tzinfo=timezone.utc) #Ajusta a UTC
        fecha_fin = datetime.strptime(fecha_fin_str, "%Y-%m-%d") #Convierte a datetime
        fecha_fin = fecha_fin.replace(tzinfo=timezone.utc) #Ajusta a UTC
        start_time = int(fecha_inicio.timestamp() * 1000) #Convierte a milisegundos
        end_time = int(fecha_fin.timestamp() * 1000) #Convierte a milisegundos
        print(f'Fecha inicio: {fecha_inicio} (UTC)')
        print(f'Fecha fin: {fecha_fin} (UTC)')
    except ValueError:
        print("Formato de fecha inválido. Use el formato 'YYYY-MM-DD'. ")
        exit()

    # Crea DataFrames vacíos para almacenar los datos descargados y procesados
    df_total = pd.DataFrame()
    df_original = pd.DataFrame()
    print("Data frames para almacenar los datos descargados df_total creados")

    # Define el número de iteraciones para descargar datos en bloques
    iteraciones = config.get("datos", {}).get("iteraciones")  # Cada iteración descarga hasta 1500 filas de datos

    for i in range(iteraciones):  # Ciclo para realizar múltiples solicitudes a la API
        print("Variables:")
        print(f'Símbolo: {symbol}')
        print(f'Intervalo: {timeframe}')
        print(f'Fecha de inicio(UTC): {datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)}')
        print(f' Timestamp de inicio (ms): {start_time}')
        print(f' Timestamp de fin (ms): {end_time}')
        print(f'Límite de datos por solicitud: {limit}')

        # Construye la URL para la solicitud a la API de Binance Futures
        url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={timeframe}&startTime={start_time}&endTime={end_time}&limit={limit}'
        response = requests.get(url)  # Realiza la solicitud

        # Procesa los datos si la solicitud fue exitosa
        if response.status_code == 200:
            data = response.json()  # Convierte la respuesta JSON a un objeto Python
            print(f"Conexión a la API Binance Futures exitosa. Request número {i+1}/{iteraciones}")
            
            # Crea un DataFrame con las columnas originales de la API
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            
            # Concatena los datos originales en un DataFrame acumulativo
            df_original = pd.concat([df_original, df], ignore_index=True)

            # Copia los datos originales para transformarlos
            df_transformed = df.copy()

            # Convierte las marcas de tiempo UNIX a formato legible y ajusta a la zona horaria
            df_transformed['timestamp'] = pd.to_datetime(df_transformed['timestamp'], unit='ms', utc=True).dt.tz_convert(tz_mexico).dt.tz_localize(None)
            df_transformed['close_time'] = pd.to_datetime(df_transformed['close_time'], unit='ms', utc=True).dt.tz_convert(tz_mexico).dt.tz_localize(None)

            # Renombra las columnas para mayor claridad
            df_transformed.columns = ['Fecha/Hora de Apertura', 'Precio de Apertura', 'Precio Maximo', 'Precio Minimo', 'Precio de Cierre', 'Volumen', 'Fecha/Hora de Cierre', 'Volumen en Divisa de Cotización', 'Número de Operaciones', 'Volumen de Compra Base', 'Volumen de Compra Divisa de Cotización', 'Ignorar']

            # Extrae componentes de fecha y hora
            df_transformed['Mes'] = df_transformed['Fecha/Hora de Apertura'].dt.month
            df_transformed['Dia'] = df_transformed['Fecha/Hora de Apertura'].dt.day
            df_transformed['Hora'] = df_transformed['Fecha/Hora de Apertura'].dt.hour
            df_transformed['Minuto'] = df_transformed['Fecha/Hora de Apertura'].dt.minute
            df_transformed['DiaSemana'] = df_transformed['Fecha/Hora de Apertura'].dt.dayofweek + 1  # Lunes como 1, domingo como 7

            # Convierte las columnas numéricas a tipo float
            df_transformed['Precio de Apertura'] = df_transformed['Precio de Apertura'].astype(float)
            df_transformed['Precio Maximo'] = df_transformed['Precio Maximo'].astype(float)
            df_transformed['Precio Minimo'] = df_transformed['Precio Minimo'].astype(float)
            df_transformed['Precio de Cierre'] = df_transformed['Precio de Cierre'].astype(float)
            df_transformed['Volumen'] = df_transformed['Volumen'].astype(float)

            # Concatena los datos transformados en un DataFrame acumulativo
            df_total = pd.concat([df_total, df_transformed], ignore_index=True)

            df_total.drop_duplicates(subset=['Fecha/Hora de Apertura'], inplace=True)

            # Elimina columnas innecesarias
            df_total.drop(columns=['Ignorar', 'Volumen en Divisa de Cotización', 'Volumen de Compra Base', 'Volumen de Compra Divisa de Cotización', 'Número de Operaciones'], inplace=True)
            print(df_total)

            # Actualiza el tiempo de inicio para la siguiente solicitud
            if not df_transformed.empty:
                fechaHoraUltimaVela = df_transformed['Fecha/Hora de Cierre'].iloc[-1]
                start_time = int(fechaHoraUltimaVela.timestamp() * 1000) #Convierte a milisegundos
                print (f"Nuevo start_time establecido: {datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)} (UTC)")
            else:
                print("Advertencia: No se recibieron datos nuevos en esta iteración. Verifica el rango de fechas. ")
                break #Detener bucle si no hay datos nuevos
        elif response.status_code != 200:
            data = response.json()
            print(data)

    # Reordena las columnas del DataFrame final
    df_total = df_total[['Mes', 'Dia', 'Hora', 'Minuto', 'DiaSemana', 'Fecha/Hora de Apertura', 'Precio de Apertura', 'Precio Maximo', 'Precio Minimo', 'Precio de Cierre', 'Fecha/Hora de Cierre', 'Volumen']]

    # # Calcula las bandas de Envelopes
    # df_total = calculate_envelopes(df_total, column='Precio de Cierre', ma_period=21, percentage=61.8)
    # df_total.rename(columns={'Envelope_Upper': 'Envelope_Upper_21_61.8', 'Envelope_Lower': 'Envelope_Lower_21_61.8'}, inplace=True)

    # df_total = calculate_envelopes(df_total, column='Precio de Cierre', ma_period=55, percentage=50)    
    # df_total.rename(columns={'Envelope_Upper': 'Envelope_Upper_55_50', 'Envelope_Lower': 'Envelope_Lower_55_50'}, inplace=True)

    df_total = calculate_envelopes(df_total, column='Precio de Cierre', ma_period=config.get("datos", {}).get("ma_period_618"), percentage=config.get("datos", {}).get("ma_percentage_618"))
    df_total.rename(columns={'Envelope_Upper': 'Envelope_Upper_21_6.18', 'Envelope_Lower': 'Envelope_Lower_21_6.18'}, inplace=True)

    df_total = calculate_envelopes(df_total, column='Precio de Cierre', ma_period=config.get("datos", {}).get("ma_period_55"), percentage=config.get("datos", {}).get("ma_percentage_55"))    
    df_total.rename(columns={'Envelope_Upper': 'Envelope_Upper_55_5', 'Envelope_Lower': 'Envelope_Lower_55_5'}, inplace=True)

    ## Una vez calculados los envelopes, detectar cruces
    # Calcular diferencias de envelopes
#     diff_upper = df_total['Envelope_Upper_21_6.18'] - df_total['Envelope_Upper_55_5']
#     diff_lower = df_total['Envelope_Lower_21_6.18'] - df_total['Envelope_Lower_55_5']

#    # Detectar cruces
#     cruces_upper = np.where((diff_upper.shift(1) < 0) & (diff_upper > 0))[0]
#     cruces_lower = np.where((diff_lower.shift(1) > 0) & (diff_lower < 0))[0]

#     indices_upper = np.unique(cruces_upper)
#     indices_lower = np.unique(cruces_lower)

#     # Dataframe de cruces
#     cruces = pd.DataFrame(index=df_total.index)
#     cruces['Cruces Upper'] = np.nan
#     cruces['Cruces Lower'] = np.nan

#     cruces.loc[indices_upper, 'Cruces_Upper'] = df_total.loc[indices_upper, 'Envelope_Upper_21_6.18']
#     cruces.loc[indices_lower, 'Cruces_Lower'] = df_total.loc[indices_lower, 'Envelope_Lower_21_6.18']

#     df_total['Cruces_Upper'] = cruces['Cruces_Upper']
#     df_total['Cruces_Lower'] = cruces['Cruces_Lower']

    ### RSI ###
    rsi_periods = config.get("common", {}).get("rsi_period_lista")
    for period in rsi_periods:
        df_total = calculate_rsi(df_total, 'Precio de Cierre', period)

    ### SMA ###
    sma_periods = config.get("common", {}).get("sma_period_lista")
    for period in sma_periods:
        df_total[f'SMA_{period}'] = df_total['Precio de Cierre'].rolling(window=period).mean()

    ### HMA ###
    hma_periods = config.get("common", {}).get("hma_period_lista")
    for period in hma_periods:
        df_total[f'HMA_{period}'] = calculate_hma(df_total, 'Precio de Cierre', period)

    ### EMA ###
    ema_periods = config.get("common", {}).get("ema_period_lista")
    for period in ema_periods:
        df_total[f'EMA_{period}'] = df_total['Precio de Cierre'].ewm(span=period, adjust=False).mean()

    ### VOL_SMA ###
    vol_sma_periods = config.get("common", {}).get("vol_sma_period_lista")
    for period in vol_sma_periods:
        df_total[f'VOL_SMA_{period}'] = df_total['Volumen'].ewm(span=period, adjust=False).mean()
    
    print ("Indicadores RSI. SMA, EMA, HMA y VOL_SMA calculados para todos los periodos configurados.")

    ### ATR, ADX, ROC, y OBV ###

    df_total = calculate_rsi(df_total, 'Precio de Cierre', period=14)
    df_total = calculate_atr(df_total, 'Precio Maximo', 'Precio Minimo', 'Precio de Cierre', period= 14)
    df_total = calculate_adx(df_total, 'Precio Maximo', 'Precio Minimo', 'Precio de Cierre', period= 14)
    df_total = calculate_roc(df_total, 'Precio de Cierre', period=10)
    df_total = calculate_obv(df_total, 'Precio de Cierre', 'Volumen')

    print("Indicadores ATR, ADX, ROC y OBV calculados correctamente")

    ### Indicadores adicionales ###
    df_total = calculate_macd(df_total, 'Precio de Cierre')
    df_total = calculate_cmf(df_total, 'Precio Maximo', 'Precio Minimo', 'Precio de Cierre', 'Volumen')
    df_total = calculate_stochrsi(df_total, 'Precio de Cierre')
    df_total = calculate_bb_width(df_total, 'Precio de Cierre')

    print("Indicadores MACD, CMF, Stochastic RSI y Bollinger Band Width calculados correctamente")

    return df_total, df_original

# Funciones auxiliares para calcular indicadores técnicos y características de las velas

# Determina el color de la vela: 1 (verde), -1 (roja), 0 (neutra)
def obtenerColordeVela(df_total):
    diferencia = df_total['Precio de Cierre'] - df_total['Precio de Apertura']
    color = np.where(diferencia > 0, 1, np.where(diferencia < 0, -1, 0))
    return color

# Calcula la longitud de la mecha superior como porcentaje del rango total de la vela
def obtenerMechaAlta(df_total):
    rangoTotalVela = df_total['Precio Maximo'] - df_total['Precio Minimo']
    dif_aperturacierre = df_total['Precio de Cierre'] - df_total['Precio de Apertura']
    mecha = ((df_total['Precio Maximo'] - df_total['Precio de Cierre']) / rangoTotalVela) * 100
    mecha_alternativa = ((df_total['Precio Maximo'] - df_total['Precio de Apertura']) / rangoTotalVela) * 100
    mecha = mecha.where(dif_aperturacierre >= 0, mecha_alternativa).fillna(0)
    return mecha

# Calcula la longitud de la mecha inferior como porcentaje del rango total de la vela
def obtenerMechaBaja(df_total):
    rangoTotalVela = df_total['Precio Maximo'] - df_total['Precio Minimo']
    dif_aperturacierre = df_total['Precio de Cierre'] - df_total['Precio de Apertura']
    mecha = ((df_total['Precio de Cierre'] - df_total['Precio Minimo']) / rangoTotalVela) * 100
    mecha_alternativa = ((df_total['Precio de Apertura'] - df_total['Precio Minimo']) / rangoTotalVela) * 100
    mecha = mecha.where(dif_aperturacierre <= 0, mecha_alternativa).fillna(0)
    return mecha

# Calcula el tamaño del cuerpo de la vela como porcentaje del rango total
def crearCuerpoVela(df_total):
    dif_aperturacierre = df_total['Precio de Cierre'] - df_total['Precio de Apertura']
    rangoTotalVela = df_total['Precio Maximo'] - df_total['Precio Minimo']
    cuerpoverde = ((df_total['Precio de Cierre'] - df_total['Precio de Apertura']) / rangoTotalVela) * 100
    cuerporojo = ((df_total['Precio de Apertura'] - df_total['Precio de Cierre']) / rangoTotalVela) * 100
    cuerpoverde = cuerpoverde.where(dif_aperturacierre >= 0, cuerporojo).fillna(0)
    return cuerpoverde

### CALCULAR RSI ###

def calculate_rsi(data, column, period=14):
    data[f'RSI_{period}'] = ta.rsi(data[column], length= period)
    return data

### CALCULATE_ATR ###
def calculate_atr(data, high_col, low_col, close_col, period=14):
    data[f'ATR_{period}'] = ta.atr(data[high_col], data[low_col], data[close_col], length=period)
    return data

### CALCULATE_ADX ###
def calculate_adx(data, high_col, low_col, close_col, period=14):
    adx = ta.adx(data[high_col], data[low_col], data[close_col], length=period)
    data[f'ADX_{period}'] = adx[f'ADX_{period}']
    return data

### CALCULATE_ROC ###
def calculate_roc(data, close_col, period=10):
    data[f'ROC_{period}'] = ta.roc(data[close_col], length=period)
    return data

### CALCULATE OBV ###
def calculate_obv(data, close_col, volume_col):
    data['OBV'] = ta.obv(data[close_col], data[volume_col])
    return data

### CALCULATE MACD ###
def calculate_macd(data, close_col, fast=12, slow=26, signal=9):
    macd = ta.macd(data[close_col], fast=fast, slow=slow, signal=signal)
    data['MACD'] = macd[f'MACD_{fast}_{slow}_{signal}']
    data['MACD_signal'] = macd[f'MACDs_{fast}_{slow}_{signal}']
    data['MACD_hist'] = macd[f'MACDh_{fast}_{slow}_{signal}']
    return data

### CALCULATE_CMF (Chaikin Money Flow) ###
def calculate_cmf(data, high_col, low_col, close_col, volume_col, period=20):
    data[f'CMF_{period}'] = ta.cmf(
        high=data[high_col],
        low=data[low_col],
        close=data[close_col],
        volume=data[volume_col],
        length=period
    )
    return data

### CALCULATE STOCHASTIC RSI ###
def calculate_stochrsi(data, close_col, length=14, rsi_length=14, k=3, d=3):
    stochrsi = ta.stochrsi(data[close_col], length=length, rsi_length=rsi_length, k=k, d=d)
    data['StochRSI_K'] = stochrsi[f'STOCHRSIk_{length}_{rsi_length}_{k}_{d}']
    data['StochRSI_D'] = stochrsi[f'STOCHRSId_{length}_{rsi_length}_{k}_{d}']
    return data

### BALCULATE BB WIDTH (Bollinger Band Width) ###
def calculate_bb_width(data, close_col, period=20, std=2):
    bbands= ta.bbands(data[close_col], length = period, std=std)
    data[f'BB_Width_{period}'] = (bbands[f'BBU_{period}_{std}.0'] - bbands[f'BBL_{period}_{std}.0'])
    return data

def calculate_envelopes(data, column, ma_period, percentage):
    # Calculate the moving average
    data['Moving Average'] = data[column].rolling(window=ma_period).mean()
    # Calculate the upper and lower envelopes
    data["Envelope_Upper"] = data['Moving Average'] * (1 + percentage / 100)
    data["Envelope_Lower"] = data['Moving Average'] * (1 - percentage / 100)
    return data

def calculate_sma(data, columna_precio, period):
    #Calcula la media móvil simple de 200 periodos y la agrega como columna 'SMA_200' al DataFrame
    data[f'SMA_{period}'] = data[columna_precio].rolling(window=period).mean()
    return data

def WMA(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

def calculate_hma(data, column, period):
    wma_half = WMA(data[column], period // 2)
    wma_full = WMA(data[column], period)
    hma_raw = 2 * wma_half - wma_full
    hma = WMA(hma_raw, int(np.sqrt(period)))
    data[f'HMA_{period}']= hma
    return hma

# Ejemplo de uso:
# data['HMA_200'] = calculate_hma(data, 'Precio de Cierre', 200)
    


if __name__ == "__main__":
    # Se crea el DataFrame con los datos descargados y procesados
    datos, df = crearDataFrame()
    print("Datos listos")

    # #Crear una columna con números secuenciales
    # datos['Secuencia']= range(1, len(datos)+1)

    # Guarda los datos originales y procesados en archivos Excel
    df.to_csv('OriginalAPI_Data.csv', index=False)  # Datos originales de la API
    datos.to_csv('Data.csv', index=False)  # Datos procesados con indicadores técnicos