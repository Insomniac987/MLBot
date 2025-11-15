import gymnasium as gym
import numpy as np
import pandas as pd
from TradingEnv import TradingEnv
import json

# Cargar config si tu env lo usa (ya lo hace)
with open("config_dev.json", "r") as f:
    config = json.load(f)

# Carga un sample df (tienes que pasar uno real)
# Example: df = pd.read_csv("data.csv")
# Debe tener columnas: ['close', 'RSI_14', 'ADX_14', 'OBV', 'MACD_hist','CMF_20','StochRSI_K','BB_Width_20']
df = pd.read_csv("Data.csv")  # o csv
df = df.rename(columns={'Precio de Cierre': 'close'})

env = TradingEnv(df, modo_backtest=False)

N = 50
results = []
for ep in range(N):
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    results.append({
        "balance_final": info.get("balance", env.balance),
        "drawdown": info.get("drawdown", env.max_drawdown),
        "n_trades": len(info.get("episode_trades", [])) if "episode_trades" in info else len(env.trades),
        "wins": env.stats["ganadoras"],
        "losses": env.stats["perdedoras"]
    })
    print(f"Ep {ep+1}: Balance {results[-1]['balance_final']:.2f} | DD {results[-1]['drawdown']:.3f} | Trades {results[-1]['n_trades']}")

# Summary
balances = np.array([r['balance_final'] for r in results])
print("SUMMARY")
print(f"mean balance: {balances.mean():.2f} | median: {np.median(balances):.2f} | std: {balances.std():.2f}")
print(f"min balance {balances.min():.2f} | max balance {balances.max():.2f}")