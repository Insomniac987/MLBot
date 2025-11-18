# bot_real_v2.py - PRODUCCION REAL

import time
import json
import os
import pandas as pd
# import pandas_ta as ta
import pandas_ta_classic as ta
from datetime import timezone, timedelta
from binance.client import Client
from binance.enums import *
from datetime import datetime
from stable_baselines3 import DQN
from decimal import Decimal, ROUND_DOWN
import numpy as np

### CONFIGURACION ###

with open("config_prod.json", "r") as f:
    config = json.load(f)

with open(os.path.expanduser("~/.mykeys/mykeys.json")) as k:
    mykeys = json.load(k)

API_KEY = mykeys.get("Clave_API")
API_SECRET = mykeys.get("Clave_secreta")
SYMBOL = config.get("symbol")
apalancamiento = config.get("apalancamiento")
SL_PORCENTAJE = config.get("sl_pct")
MONEDA = SYMBOL.replace("USDT", "")
WINDOW_SIZE = config.get("window_size")
TIMEFRAME = config.get("timeframe")
FACTOR_SEGURIDAD = config.get("factor_seguridad")
EPISODE_STEPS = config.get("episode_steps")

match apalancamiento:
  case "1x":
      LEVERAGE = 1
  case "5x":
      LEVERAGE = 5
  case "10x":
      LEVERAGE = 10
  case "25x":
      LEVERAGE = 25
  case "50x":
      LEVERAGE = 50
  case "100x":
      LEVERAGE = 100
  case _:
      print("Opción no válida")

# === Inicializar cliente ===
client = Client(API_KEY, API_SECRET)
client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
print(f"✅ Apalancamiento seteado a {LEVERAGE}x para {SYMBOL}")

# === Funciones ===

def get_lot_step(symbol=SYMBOL):
    info = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    return float(f['stepSize'])
    return 0.001

def ajustar_cantidad(cantidad, step=get_lot_step(SYMBOL)):
    return float((Decimal(cantidad).quantize(Decimal(str(step)), rounding=ROUND_DOWN)))

### CALCULATE RSI ###
def calculate_rsi(data, column, period=14):
    data[f'RSI_{period}'] = ta.rsi(data[column], length= period)
    return data

### CALCULATE_ADX ###
def calculate_adx(data, high_col, low_col, close_col, period=14):
    adx = ta.adx(data[high_col], data[low_col], data[close_col], length=period)
    data[f'ADX_{period}'] = adx[f'ADX_{period}']
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

def obtener_ultimas(symbol=SYMBOL, interval=TIMEFRAME, limit=WINDOW_SIZE + 50):

    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df_total = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Casting

    df_total["open"] = df_total["open"].astype(float)
    df_total["low"] = df_total["low"].astype(float)
    df_total["high"] = df_total["high"].astype(float)
    df_total["close"] = df_total["close"].astype(float)
    df_total["volume"] = df_total["volume"].astype(float)
    df_total["timestamp"] = df_total["timestamp"].astype(int)

    # Indicators

    # calculate RSI
    df_total = calculate_rsi(df_total, 'close', period=14)
    # calculate ADX
    df_total = calculate_adx(df_total, 'high', 'low', 'close', period= 14)
    # calculate OBV
    df_total = calculate_obv(df_total, 'close', 'volume')
    #calculate MACD
    df_total = calculate_macd(df_total, 'close')
    #calculate cmf
    df_total = calculate_cmf(df_total, 'high', 'low', 'close', 'volume')
    #calculate stochrsi
    df_total = calculate_stochrsi(df_total, 'close')
    #calculate bb width
    df_total = calculate_bb_width(df_total, 'close')

    df_total = df_total.dropna().reset_index(drop=True)
    df_total.to_csv('Data_prod.csv', index= False)

    return df_total

def get_state(df, balance_norm, cur_pct, equity_change, trade_duration, drawdown, pos_vector):
    window = df.iloc[-WINDOW_SIZE:]
    state_array = window[['close', 'RSI_14', 'ADX_14', 'OBV', 'MACD_hist',
                          'CMF_20', 'StochRSI_K', 'BB_Width_20']].values
    
    # Normalize
    means = state_array.mean(axis=0)
    stds = state_array.std(axis=0) + 1e-8
    obs = ((state_array - means) / stds).flatten()

    # Build agent state vector
    state_vector = np.array([
        balance_norm,
        cur_pct,
        equity_change,
        trade_duration,
        drawdown] + pos_vector,
        dtype=np.float32)
    
    full_obs = np.concatenate([obs.astype(np.float32), state_vector]).astype(np.float32)

    return full_obs


# obtain symbol
_symbol_info_cache = None
def _get_symbol_info():
    global _symbol_info_cache
    if _symbol_info_cache is None:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == SYMBOL:
                _symbol_info_cache = s
                break
    return _symbol_info_cache

# Get tick size
def get_tick_size(symbol=SYMBOL):
    s = _get_symbol_info()
    for f in s['filters']:
        if f['filterType'] == 'PRICE_FILTER':
            return float(f['tickSize'])
    return 0.01

def round_price(x, tick):
    #round down to tick
    return float(Decimal(x).quantize(Decimal(str(tick)), rounding=ROUND_DOWN))

def cancel_all_user_orders(symbol=SYMBOL):
    try:
        client.futures_cancel_all_open_orders(symbol=SYMBOL)
    except Exception as e:
        print("Warning cancelling orders:", e)

def close_operation(posicion_abierta):
    try:
        side = SIDE_SELL if float(posicion_abierta['positionAmt']) > 0 else SIDE_BUY
        quantity = abs(float(posicion_abierta['positionAmt']))
        quantity = ajustar_cantidad(quantity, step=get_lot_step(SYMBOL)) # para BTCUSDT

        client.futures_create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity,
            reduceOnly=True
        )
        print(f"❌ Position {side} CLOSED, quantity: {quantity}")
    except Exception as e:
        print(f"Error while closing position {e}")

def ejecutar_operacion(action, price):

    #Obtener el precio actual
    precio_actual = float(client.get_symbol_ticker(symbol=SYMBOL)['price'])
    #Obtener entry price
    entry_price = None

    ### BALANCE ###
    futures_balances = client.futures_account_balance()
    usdt_balance = next((item for item in futures_balances if item['asset'] == 'USDT'), None)
    available_balance = float(usdt_balance['availableBalance'])
    #Calcular la cantidad máxima de BTC que se puede comprar
    cantidad = (available_balance * FACTOR_SEGURIDAD * LEVERAGE) / precio_actual
    #Redondear a la cantidad mínima permitida (BTCUSDT: 0.001)
    cantidad = ajustar_cantidad(cantidad, step=get_lot_step(SYMBOL))

    if usdt_balance:
       print(f"\033[96m")
       print(f"Saldo en USDT en Futuros: {usdt_balance}")
       print(f"Balance disponible para trading: {available_balance}")
       print(f"Cantidad de BTC disponible: {cantidad}\033[0m")

    # detect if position is open

    info_pos = client.futures_position_information(symbol=SYMBOL)
    posicion_abierta = next(
    (p for p in info_pos if float(p['positionAmt']) != 0),None
    )
    
    # open long
    if action == 1:
        if not posicion_abierta:
            try:
                # place market entry
                order = client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=cantidad
                )
                print(f"📈 Open LONG order sent, Action: {action}")

            except Exception as e:
                print(f"Error opening long market order: {e}")
                return

            # small sleep then fetch position info to confirm entry proce & qty
            time.sleep(0.5)
            info_pos = client.futures_position_information(symbol=SYMBOL)
            pos = next((p for p in info_pos if float(p['positionAmt']) != 0), None)
            if not pos:
                print("Warning: position not found after opening. Skipping SL placement.")
            else:
                entry_price = float(pos.get('entryPrice') or precio_actual)
                amt = abs(float(pos['positionAmt']))
                # delay before cancelling orders
                time.sleep(0.3)
                # cancel previous orders (optional but recommended)
                cancel_all_user_orders(SYMBOL)
                # 4) calc stop price and round to tick
                tick = get_tick_size(SYMBOL)
                sl_real = SL_PORCENTAJE / LEVERAGE
                sl_price = round_price(entry_price * (1 - sl_real), tick)
                try:
                    client.futures_create_order(
                        symbol=SYMBOL,
                        side=SIDE_SELL,             # to close a LONG -> SELL
                        type='STOP_MARKET',
                        stopPrice=str(sl_price),
                        closePosition=True,
                        priceProtect=True
                    )
                    print(f"✅ SL set at {sl_price} for LONG")
                except Exception as e:
                    print("Error placing SL for long:", e)
        elif posicion_abierta and float(posicion_abierta["positionAmt"]) < 0:
            print(f"NOOP: cannot open LONG while SHORT, Action: {action}")
            return
        else:
            print(f"NOOP: already in position, Action {action}")
    # close long
    elif action == 2:
        if posicion_abierta:
            cancel_all_user_orders(SYMBOL)
            close_operation(posicion_abierta)
            wait_next_candle = True
        else:
            print(f"NOOP: Invalid Action, there's no LONG to close, Action: {action}")

    # open short
    elif action == 3:
        if not posicion_abierta:
            try:
                # place market entry
                order = client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=cantidad
                )
                print(f"📉 Open SHORT order sent, Action: {action}")
            except Exception as e:
                print("Error opening short market order:", e)
                return
            # small sleep then fetch position info to confirm entry proce & qty
            time.sleep(0.5)
            info_pos = client.futures_position_information(symbol=SYMBOL)
            pos = next((p for p in info_pos if float(p['positionAmt']) != 0), None)
            if not pos:
                print("Warning: position not found after opening short. Skipping SL placement.")
            else:
                entry_price = float(pos.get('entryPrice') or precio_actual)
                amt = abs(float(pos['positionAmt']))
                # delay before cancelling orders
                time.sleep(0.3)
                # cancel previous orders (optional but recommended)
                cancel_all_user_orders(SYMBOL)
                tick = get_tick_size(SYMBOL)
                sl_real = SL_PORCENTAJE / LEVERAGE
                sl_price = round_price(entry_price * (1 + sl_real), tick)
                try:
                    client.futures_create_order(
                        symbol=SYMBOL,
                        side=SIDE_BUY,              # to close a SHORT -> BUY
                        type='STOP_MARKET',
                        stopPrice=str(sl_price),
                        closePosition=True,
                        priceProtect=True
                    )
                    print(f"✅ SL set at {sl_price} for SHORT")
                except Exception as e:
                    print("Error placing SL for short:", e)

        elif posicion_abierta and float(posicion_abierta["positionAmt"]) > 0:
            print(f"NOOP: cannot open SHORT while LONG, Action: {action}")
            return
        elif posicion_abierta:
            print(f"NOOP: already in position, Action: {action}")
    
    # close short
    elif action == 4:
        if posicion_abierta:
            cancel_all_user_orders(SYMBOL)
            close_operation(posicion_abierta)
            wait_next_candle = True
        else:
            print(f"NOOP: Invalid Action, there's no SHORT to close")
    else:
        if posicion_abierta:
            print(f"⏸️ HOLD IN POSITION, Action: {action}")
        else:
            print(f"⏸️ HOLD OUT OF POSITION, Action: {action}")

# Get initial balance only the first time
def load_initial_balance(current_balance):
    path = "initial_balance.json"

    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({"initial_balance": current_balance}, f)
        return current_balance
    
    with open(path, "r") as f:
        return json.load(f)["initial_balance"]
    
def get_futures_balance(client, asset="USDT"):
    try:
        balances = client.futures_account_balance()
        for b in balances:
            if b["asset"] == asset:
                return b
        return None
    except Exception as e:
        print(f"Error obtaining futures balance: {e}")
        return None

if __name__ == "__main__":

    # === Load trained model ===
    model = DQN.load("../BACKTESTING/expert_professional_bots/super_winner_8M_best_model")

    ### LOAD PREVIOUS STATE IF EXISTING ###

    state_path = "bot_last_state.json"
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
            last_candle = state.get("last_candle", None)
            wait_next_candle = state.get("wait_next_candle", False)
            print("📦 State restored from disk")
    else:
        last_candle = None
        wait_next_candle = False

    # === Loop de inferencia ===
    print("🚀 Starting bot in production...")

    # Variables persistentes entre iteraciones
    entry_timestamp = None   # timestamp (ms) cuando se abrió la posición detectada por el bot
    prev_pos_exists = False  # para detectar transiciones
    prev_side = None
    # Initialize binance client
    client = Client(API_KEY, API_SECRET)
    # Obtain first real balance
    futures_balance= get_futures_balance(client)
    current_balance = float(futures_balance['availableBalance'])
    # Load or create initial balance 
    initial_balance = load_initial_balance(current_balance)
    # Load peak balance from file if existing
    try:
        with open("peak_balance.json", "r") as f:
            peak_balance = float(json.load(f)["peak"])
    except:
        peak_balance = current_balance

    previous_balance = current_balance

    while True:
        try:
            df = obtener_ultimas(SYMBOL)
            current_price = df.iloc[-1]['close']
            timestamp_last_candle = int(df.iloc[-1]["timestamp"])

            # First execution -> just initializes
            if last_candle is None:
                last_candle = timestamp_last_candle
                wait_next_candle = False
                print("⏳ Bot inicializado. Esperando primera vela nueva...")
                time.sleep(5)
                continue

            # There's no new candle → do nothing
            if timestamp_last_candle == last_candle:
                time.sleep(5)
                continue

            # 🎉 New candle is here
            print("\n🟩 New candle detected:", timestamp_last_candle)
            last_candle = timestamp_last_candle

            # Detect current position
            info_pos = client.futures_position_information(symbol=SYMBOL)
            posicion_abierta = next(
                (p for p in info_pos if float(p['positionAmt']) != 0),
                None
            )
            ### === Build REAL STATE for DQN === ###

            futures_balance= get_futures_balance(client)
            current_balance = float(futures_balance['availableBalance'])
            peak_balance = max(peak_balance, current_balance)

            # Save each update of peak balance
            with open("peak_balance.json", "w") as f:
                json.dump({"peak": peak_balance}, f)

            # STATE: Calculate balance_norm
            # ****** CHECK OK ******
            balance_norm = current_balance / max(initial_balance, 1)

            # STATE: Calculate equity_balance 
            # ****** CHECK OK ******
            equity_change = (current_balance - previous_balance) / max(initial_balance, 1)
            previous_balance = current_balance

            # Detectamos si hay posición ahora
            posicion_abierta = next(
                (p for p in info_pos if float(p['positionAmt']) != 0),
                None
            )
            current_pos_exists = posicion_abierta is not None

            # Detect side and amt if existing
            if current_pos_exists:
                amt = float(posicion_abierta["positionAmt"])
                side = "LONG" if amt > 0 else "SHORT"
            else:
                amt = 0.0
                side = None

            # Transición: sin posición -> con posición (marcamos entry_timestamp)
            if current_pos_exists and not prev_pos_exists:
                # Nueva posición detectada ahora, guardamos timestamp de la vela actual como entrada
                entry_timestamp = timestamp_last_candle
                prev_side = side

            # Transición: con posición -> sin posición (limpiamos datos de entrada)
            if not current_pos_exists and prev_pos_exists:
                entry_timestamp = None
                prev_side = None

            # Calcula cur_pct, trade_duration, drawdown, pos_vector, equity_change
            if current_pos_exists:
                # entry_price puede ser "0" si Binance nunca definió, fallback a current_price
                entry_price = float(posicion_abierta.get("entryPrice") or current_price)

                # STATE: calculate cur_pct: porcentaje PnL ajustado por lado 
                # ****** CHECK OK ******
                cur_pct = (current_price - entry_price) / max(entry_price, 1e-8)
                if amt < 0: # SHORT
                    cur_pct = -cur_pct

                # STATE: calculate trade_duration in STEPS 
                # ****** CHECK OK ******
                if entry_timestamp is not None:
                    duration_minutes = (timestamp_last_candle - entry_timestamp) / 60000 # 1 minute = 60000 ms
                    duration_steps = duration_minutes / 5 # (mins in the trade / 5 min) to obtain the current steps in 5 mins candle
                    trade_duration = min(duration_steps / EPISODE_STEPS, 1.0)
                else:
                    trade_duration = 0.0

                # STATE: calculate drawdown in STEPS
                # ****** CHECK OK ******
                drawdown = (peak_balance - current_balance) / max(peak_balance, 1)
                # position one-hot
                # STATE: calculate POS: check current position
                # ****** CHECK OK ******
                pos_vector = [1, 0, 0] if side == "LONG" else [0, 1, 0]

            else:
                # no position
                cur_pct = 0.0
                trade_duration = 0.0
                drawdown = 0.0
                pos_vector = [0, 0, 1]

            # STATE logging
            print(
                f"balance_norm: {balance_norm}\n"
                f"cur_pct: {cur_pct}\n"
                f"equity_change: {equity_change}\n"
                f"trade_duration: {trade_duration}\n"
                f"drawdown: {drawdown}\n"
                f"pos: {pos_vector}\n"
                )

            # guardar estado de presencia de posición para la próxima iteración
            prev_pos_exists = current_pos_exists

            # Obtain state and predict action
            state = get_state(
                df,
                balance_norm=balance_norm,
                cur_pct=cur_pct,
                equity_change=equity_change,
                trade_duration=trade_duration,
                drawdown=drawdown,
                pos_vector=pos_vector)
            action, _ = model.predict(state, deterministic=True)

            # if action is HOLD
            if action == 0:
                # HOLD inpos
                if posicion_abierta:
                    print("⏸️ HOLD IN POSITION")
                # HOLD outpos
                else:
                    print("⏸️ HOLD OUT OF POSITION")

            elif action in [1,2,3,4]:
                ejecutar_operacion(action, current_price)

            # Guardar estado
            state_dict = {
                "last_candle": last_candle,
                "wait_next_candle": wait_next_candle,
            }
            with open(state_path, "w") as f:
                json.dump(state_dict, f)

            time.sleep(5)

        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(10)