from TradingEnv import TradingEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
import json
import os

model_path ="best_model/best_model.zip"
SEED = 42

## LEER ARCHIVO DE CONFIGURACIÓN ##
with(open("config_dev.json", "r")) as f:
   config = json.load(f)

fijar_seed = config.get("datos", {}).get("fijar_seed")

df = pd.read_csv('Data.csv')
df.rename(columns={
    'Precio de Cierre': 'close',
    'EMA_150': 'EMA_150',
    'Volumen': 'volume'
}, inplace=True)

df = df.dropna().reset_index(drop=True)

env = DummyVecEnv([lambda: TradingEnv(df)])

# QUITAR ALEATORIEDAD
if fijar_seed == True:
    env.seed(SEED)
    np.random.seed(SEED)
    env.action_space.seed(SEED)
print(f"La configuración de seed está en {fijar_seed}")

if os.path.exists(model_path):
    model = DQN.load(model_path, env=env)
    print("Best Model cargado desde disco")
else:
    print("El modelo no existe y no se puede ejecutar")
    exit()

obs = env.reset()
done = [False]
balances = []
drawdowns = []
tipos = []
last_info = None

while not done[0]:
    action, _ = model.predict(obs, deterministic= True)
    obs, reward, done, info = env.step(action)

    # Guarda el ultimo info antes del auto-reset de DummyVecEnv
    last_info = info[0]

    balances.append(info[0]['balance'])
    drawdowns.append(info[0]['drawdown'])
    tipos.append(info[0].get('trade_type', 'none'))


    print(f"Step: {info[0]['step']}, Action: {action}, Reward: {reward[0]:.2f}, Balance: {info[0]['balance']:.2f}")
    if info[0]['balance'] <= 1:
        break

raw_env = env.envs[0]  # accede al entorno base

# Balance final robusto (evita el auto-reset)
final_balance = (balances[-1] if balances else (last_info['balance'] if last_info else raw_env.balance))

# Trades del episodio: prioriza los devueltos en info al terminar
if last_info and 'episode_trades' in last_info:
    trades= last_info['episode_trades']
else:
    # Fallback ( puede estar vacío si DummyVecEnv ya auto-reseteó)
    trades = list(getattr(raw_env, 'trades', []))

initial_balance = getattr(raw_env, 'initial_balance', 0.0)

# Filtrar trades cerrados (normales o forzados)
closed_trades = [
    t for t in trades
    if any(keyword in t['type'].upper() for keyword in ['LONG CLOSE', 'SHORT CLOSE', 'FORCED CLOSE'])
]

# Ganadoras
winning_trades = [t for t in closed_trades if 'profit' in t and t['profit'] > 0]

# Perdedoras
losing_trades = [t for t in closed_trades if 'profit' in t and t['profit'] <= 0]

# Longs
longs = [
    t for t in trades 
    if 'LONG BUY' in t['type']
    ]

shorts = [
    t for t in trades
    if 'SHORT SELL' in t['type']
]

# Shorts

# Hold (acciones donde no se hizo nada, si las guardas como 'hold')
hold_inpos_count = sum(1 for tt in tipos if tt == 'hold_inpos')
hold_outpos_count = sum(1 for tt in tipos if tt == 'hold_outpos')
noop_count = sum(1 for tt in tipos if tt == 'noop')
total_actions = len(tipos)

# Cálculo de efectividad
if closed_trades:
    winrate = len(winning_trades) / len(closed_trades) * 100
else:
    winrate = 0.0

total_profit = final_balance - initial_balance
max_dd = max(drawdowns or [getattr(raw_env, 'max_drawdown', 0.0)])

# Mostrar estadísticas
print("\nResumen del backtest:")
print(f"Balance inicial: {initial_balance:.2f}")
print(f"Balance final: {final_balance:.2f}")
print(f"Ganancia neta: {total_profit:.2f}")
print(f"Drawdown máximo: {max_dd:.2f}")
print(f"Total de trades cerrados: {len(closed_trades)}")
print(f" - Ganadores: {len(winning_trades)}")
print(f" - Perdedoras: {len(losing_trades)}")
print(f" - LONGS: {len(longs)}")
print(f" - SHORTS: {len(shorts)}")
print(f"Operaciones HOLD in position {hold_inpos_count}")
print(f"Operaciones HOLD out of position {hold_outpos_count}")
print(f"Operaciones No-Op: {noop_count}")

print(f"Efectividad (winrate): {winrate:.2f}%")
if total_actions > 0:
    print(f"% Operaciones HOLD in position: {hold_inpos_count/total_actions:.2%}")
    print(f"% Operaciones HOLD out of position {hold_outpos_count/total_actions:.2%}")
    print(f"% Operaciones NO-OP: {noop_count/total_actions:.2%}")
else:
    print("% Operaciones HOLD: 0.00%")
    print("% Operaciones No-Op: 0.00%")

# ## Exportar mmétricas a CSV ##
# import datetime
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# summary = {
#     "balance_inicial": initial_balance,
#     "balance_final": final_balance,
#     "ganancia_neta": total_profit,
#     "max_drawdown": max_dd,
#     "trades_totales": len(closed_trades),
#     "winrate": winrate,
# }
# # pd.DataFrame([summary]).to_csv(f"backtest_summary_{timestamp}.csv", index=False)
env.close()