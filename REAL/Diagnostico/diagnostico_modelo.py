import json
import os
import pandas as pd
import numpy as np
from binance.client import Client
from stable_baselines3 import DQN
from datetime import timezone, timedelta, datetime
import time

### CARGAR CONFIGURACIÓN Y LLAVES ###

with open("../config_prod.json", "r") as f:
    config = json.load(f)
with open(os.path.expanduser("~/.mykeys/mykeys.json")) as k:
    mykeys = json.load(k)

API_KEY = mykeys.get("Clave_API")
API_SECRET = mykeys.get("Clave_Secreta")
SYMBOL = config.get("symbol")
apalancamiento = config.get("apalancamiento")
TIMEFRAME = config.get("timeframe")
WINDOW_SIZE = config.get("window_size")

### ENVELOPE PARAMS ###
ma_period_618 = config.get("ma_period_618")
ma_period_55 = config.get("ma_period_55")
ma_percentage_618 = config.get("ma_percentage_618")
ma_percentage_55 = config.get("ma_percentage_55")

### CLIENTE BINANCE ###
client = Client(API_KEY, API_SECRET)

### FUNCIONES ###
def calculate_envelopes(df, column, ma_period, percentage):
    df["Moving Average"] = df[column].rolling(window=ma_period).mean()
    df["Envelope_Upper"] = df["Moving Average"] * (1 + percentage / 100)
    df["Envelope_Lower"] = df["Moving Average"] * (1 - percentage / 100)
    return df

def obtener_datos(symbol, interval=TIMEFRAME, dias_atras=90):
    ms_por_vela = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000
    }

    limite = 1000
    now = int(time.time() * 1000)
    intervalo_ms = ms_por_vela[interval]
    inicio = int((datetime.now(timezone.utc)- timedelta(days=dias_atras)).timestamp() * 1000)

    df_final = pd.DataFrame()

    ### MIENTRAS HAYA FECHA ANTERIOR A HOY ###
    while inicio < now:
        klines = client.get_historical_klines(symbol, interval, start_str=inicio, limit=limite)
        if not klines:
            break
        df_temp = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df_final = pd.concat([df_final, df_temp], ignore_index=True)
        inicio= int(df_temp.iloc[-1]["timestamp"]) + intervalo_ms
        time.sleep(0.01)
    df_final["close"] = df_final["close"].astype(float)
    df_final["volume"] = df_final["volume"].astype(float)
    df_final["EMA_150"] = df_final["close"].ewm(span=150).mean()

    df_final = calculate_envelopes(df_final, 'close', ma_period_618, ma_percentage_618)
    df_final.rename(columns={"Envelope_Upper": "Envelope_Upper_21_6.18",
                    "Envelope_Lower": "Envelope_Lower_21_6.18"}, inplace=True)
    
    df_final = calculate_envelopes(df_final, 'close', ma_period_55, ma_percentage_55)
    df_final.rename(columns={"Envelope_Upper": "Envelope_Upper_55_5",
                             "Envelope_Lower": "Envelope_Lower_55_5"}, inplace=True)
    
    df_final = df_final.dropna().reset_index(drop=True)
    return df_final

def get_state(df):
    state = df.iloc[-WINDOW_SIZE:]
    state_array = state[["close", "volume", "EMA_150", "Envelope_Upper_21_6.18", "Envelope_Lower_21_6.18", "Envelope_Upper_55_5", "Envelope_Lower_55_5"]].values
    # Normalización igual que en el training
    state_array = (state_array - np.mean(state_array, axis=0)) / (np.std(state_array, axis=0) + 1e-8)
    return np.expand_dims(state_array, axis=0).astype(np.float32)

### EJECUTAR DIAGNÓSTICO ###
df = obtener_datos(SYMBOL)

print("Cargando modelo...")
model = DQN.load("../BACKTESTING/expert_professional_bots/coke_SHORTER_8M")

print("Generando estado...")
state = get_state(df)

print("Ejecutando predicción...")
action, _ = model.predict(state, deterministic = True)

acciones_map = {0: "HOLD", 1: "LONG", 2: "SHORT"}
print(f"Acción recomendada por el modelo: {action[0]} ({acciones_map[int(action[0])]})")
    

