import pandas as pd
import os
import numpy as np
from TradingEnv import TradingEnv
from datetime import datetime
import json
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

def fmt(x):
   try:
      return f"{x:.2e}" if abs(x) > 1e6 else f"{x:.2f}"
   except Exception:
      return str(x)

model_path ="model.zip"

SEED = 42

### CONFIGURAR JSON DE VARIABLES ###

with(open("config_dev.json", "r")) as f:
   config = json.load(f)

fijar_seed = config.get("datos", {}).get("fijar_seed")

### Establecer el total timesteps ###
train_steps = config.get("datos", {}).get("train_steps")
apalancamiento = config.get("common", {}).get("apalancamiento")

match apalancamiento:
  case "1x":
      leverage = 1
  case "5x":
      leverage = 5
  case "10x":
      leverage = 10
  case "25x":
      leverage = 25
  case "50x":
      leverage = 50
  case "100x":
      leverage = 100
  case _:
      print("Opción no válida")

df = pd.read_csv("Data.csv")
df.rename(columns={
    'Precio de Cierre': 'close',
    'EMA_150': 'EMA_150',
    'Volumen': 'volume'
}, inplace=True)

df = df.dropna().reset_index(drop=True)

### Validaciones del dataframe ###

window_size = config.get("datos", {}).get("window_size")
episode_steps = config.get("datos", {}).get("episode_steps")
required_columns = ['close', 'RSI_14', 'ADX_14', 'OBV', 'MACD_hist', 'CMF_20', 'StochRSI_K', 'BB_Width_20']
#validación de correlación
correlation_matrix = df[required_columns].corr()
print(correlation_matrix)

if (abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix, 1)]) > 0.95).any():
   print("⚠️ Warning: Hay indicadores muy correlacionados, podrías reducir features.")

assert all(col in df.columns for col in required_columns), f"Columnas faltantes: {[col for col in required_columns if col not in df.columns]}"
assert len(df) > window_size + episode_steps, f"Dataset demasiado pequeño: {len(df)} filas, pero necesitas al menos {window_size + episode_steps + 1}"

print("✅ Todas las validaciones pasaron - Creando entornos...")

# ✅ FORMA CORRECTA - misma instancia
# Usar la misma instancia siempre
env = DummyVecEnv([lambda: TradingEnv(df)])
eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])

# QUITAR ALEATORIEDAD
if fijar_seed:
    env.seed(SEED)
    eval_env.seed(SEED)
    env.action_space.seed(SEED)
    np.random.seed(SEED)

print(f"La configuración de seed está en {fijar_seed}")

if os.path.exists(model_path):
    # Si ya existe el modelo cargarlo
    print("##########################################################")
    print("🧠 Modelo existente detectado. Reanudando entrenamiento...")
    print("##########################################################")
    model = DQN.load(model_path, env=env, print_system_info=True)

else:
    # Si no existe, crear el modelo DQN
    model = DQN(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=2.5e-4, 
        buffer_size=100000,
        learning_starts=10000, 
        batch_size=128,
        tau=0.01,
        gamma=0.99,
        exploration_fraction=0.4, # AJUSTE ACEPTADO
        exploration_final_eps=0.05,
        target_update_interval=5000, # CAMBIAR A UN VALOR MÁS ALTO (MÁS ESTABLE)
        train_freq=4,
        gradient_steps=4,
        policy_kwargs=dict(net_arch=[256, 256])
    )
    print("Modelo nuevo creado")

# Logger para consola + archivo
new_logger = configure("logs", ["stdout", "csv", "tensorboard"])
model.set_logger(new_logger)

# EvalCallback para evaluación periódica
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model',
    log_path='./logs/',
    eval_freq=50000,
    n_eval_episodes=3,
    deterministic=True,
    render=False
)

###########################
### FASE DE APRENDIZAJE ###
###########################

print(f"🏋️ Entrenando modelo por {train_steps:,} pasos adicionales...")

model.learn(total_timesteps=train_steps, callback=eval_callback, log_interval=10)

# Guardar el modelo entrenado
model.save(model_path)
print("✅ Modelo guardado correctamente.")

###########################
####### BACKTESTING #######
###########################

backtest_env = TradingEnv(df, modo_backtest=True)
obs, _ = backtest_env.reset()
done = False
balances, drawdowns, tipos, trade_lines = [], [], [], []
episode_trades = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = backtest_env.step(action)
    done = terminated or truncated
    balances.append(info['balance'])
    drawdowns.append(info['drawdown'])
    tipos.append(info.get('trade_type', 'none'))
    trade_lines.append(
        f"Step: {info['step']}, Tipo_trade: {info['trade_type']},"
        f"Action: {action}, Result: {info.get('result')} Reward: {reward:.2f},"
        f"Balance: {info['balance']:.2f}, Drawdown: {info['drawdown']:.2f},"
        f"Profit: {info['profit']:.2f}"
    )
    print(trade_lines[-1])
    if done and 'episode_trades' in info:
        episode_trades = info['episode_trades']

###########################
####### RESULTADOS ########
###########################

## CALCULAR EFICIENCIA DE LA ESTRATEGIA"

# print(f"Total de operaciones hold (no hacer nada): {num_holds}")
print(f"\nBalance final: {balances[-1]:.2f}")
print(f"Drawdown máximo: {max(drawdowns):.2f}")
print(f"Balance máximo alcanzado: {max(balances):.2f}")
print(f"Balance mínimo alcanzado: {min(balances):.2f}")

# Usar el snapshot estructurado del episodio
trades = episode_trades if episode_trades else list(backtest_env.trades)

### Estadísticas más robustas ###
close_types = {'LONG CLOSE','SHORT CLOSE', 'FORCED CLOSE'}
buy_types = {'LONG BUY', 'SHORT SELL'}
total_profit_trades = 0
ganadoras = 0
perdedoras = 0
forzadas = 0
aperturas = 0

holds_inpos = sum(1 for tt in tipos if tt == 'hold_inpos')
holds_outpos = sum(1 for tt in tipos if tt == 'hold_outpos')
noops = sum(1 for tt in tipos if tt == 'noop')


for t in trades:
    ttype = t.get('type')
    if ttype in buy_types:
       aperturas += 1
    if ttype in close_types:
       p = t.get('profit')
       if p is not None and np.isfinite(p):
          total_profit_trades += 1
          if p > 0:
             ganadoras += 1
          else:
             perdedoras += 1
       if ttype == 'FORCED CLOSE' or t.get('forced'):
          forzadas += 1

# Salidas finales segurdas
final_balance = balances[-1] if balances else backtest_env.balance
max_dd = max(drawdowns) if drawdowns else getattr(backtest_env, 'max_drawdown', 0.0)
max_bal = max(balances) if balances else final_balance
min_bal = min(balances) if balances else final_balance

print(f"\n📊 ESTADÍSTICAS DE TRADING CORREGIDAS:")
print(f"Total de operaciones: {total_profit_trades}")
print(f"Ganadoras: {ganadoras}")
print(f"Perdedoras: {perdedoras}")
print(f"Forzadas: {forzadas}")
print(f"Aperturas: {aperturas}")
print(f"Holds in position: {holds_inpos}")
print(f"Holds out if position {holds_outpos}")
print(f"Noops: {noops}")

### GUARDAR RESULTADOS EN ARCHIVO DE LOG ###

output_resultados = "ml_learning"
os.makedirs(output_resultados, exist_ok=True)

timestamp = datetime.now().strftime("%m%d%H%M")
filename = f"LEV:{leverage}_TOTOPS:{total_profit_trades}_HOLDSINPOS:{holds_inpos}_HOLDSOUTPOS:{holds_outpos}_NOOPS:{noops}_WIN:{ganadoras}_LOSE:{perdedoras}_STEPS:{train_steps}_DATE:{timestamp}_BALFINAL:{fmt(final_balance)}_DWDMAX:{fmt(max_dd)}_MAXBAL:{fmt(max_bal)}_MINBAL:{fmt(min_bal)}"


with open(os.path.join(output_resultados, filename), 'a', buffering=1) as f:
   
   f.write("RESULTADOS\n")
   f.write(f"\nBalance final: {final_balance}\n")
   f.write(f"Drawdown máximo: {max_dd:.2f}\n")
   f.write(f"Balance máximo alcanzado: {max_bal:.2f}\n")
   f.write(f"Balance mínimo alcanzado: {min_bal:.2f}\n")


   ## Insertar el contenido de TradingEnv.py ##
   f.write("CONTENIDO DE TradingEnv.py --- \n\n")
   with open("TradingEnv.py", "r") as env_file:
      f.write(env_file.read())
   
   ##Guardar los detalles de cada trade
   f.write("\n\nTRADES:\n")
   f.write('\n'.join(trade_lines))

try:
  env.close()
except:
    pass

try:
  eval_env.close()
except:
   pass
