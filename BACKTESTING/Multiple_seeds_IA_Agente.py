import pandas as pd
import os
from TradingEnv import TradingEnv
from datetime import datetime
import json
import time
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

### CONFIGURAR JSON DE VARIABLES ###

with(open("config_dev.json", "r")) as f:
   config = json.load(f)

## Cargar datos ##

df = pd.read_csv("Data.csv")
df.rename(columns={
    'Precio de Cierre': 'close',
    'EMA_150': 'EMA_150',
    'Volumen': 'volume'
}, inplace=True)

df = df.dropna().reset_index(drop=True)

fijar_seed = config.get("datos", {}).get("fijar_seed")

### Parámetros ###

steps = config.get("datos", {}).get("steps")
base_path = "multi_model_runs"
os.makedirs(base_path, exist_ok=True)
# Número de bots a entrenar #
N = config.get("datos", {}).get("seeds_number")

# Lista para guardar resultados #
results = []

### ENTRENAR N BOTS CON DIFERENTES SEEDS ###

for i in range(N):

   seed = i + 1
   print(f"\n🤖 Entrenando bot {i+1} con SEED = {seed}")

   ## Entorno de entrenamiento y evaluación ##
   env = DummyVecEnv([lambda: TradingEnv(df)])
   eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])

   env.seed(seed)
   eval_env.seed(seed)

   # Crear modelo o cargarlo

   initial_model_path =f"model_seed_{seed}.zip"

   if os.path.exists(initial_model_path):
     # Si ya existe el modelo cargarlo
     model = DQN.load(initial_model_path, env=env)
     print("Modelo cargado desde disco")
   else:
     # Si no existe, crear el modelo DQN
     model = DQN('MlpPolicy', 
                 env,
                 verbose=1,
                 learning_rate=1e-4,
                 buffer_size=5000, 
                 learning_starts=1000,
                 batch_size=32,
                 tau=0.1)
     print("Modelo nuevo creado")
   # Callback para guardar el mejor modelo según reward
   eval_callback = EvalCallback(
      eval_env,
      best_model_save_path=f"{base_path}/model_{i+1}_seed{seed}",
      log_path=f"{base_path}/logs_{i+1}",
      n_eval_episodes= 2,
      eval_freq = 1000,
      deterministic = True,
      render = False
   )
   
   ###########################
   ### FASE DE APRENDIZAJE ###
   ###########################

   model.learn(total_timesteps=steps, callback=eval_callback, log_interval=10)

   # Guardar el modelo entrenado
   model.save(f"model_seed_{seed}")

   # Evaluar el best_model después de entrenar
   best_model_path = f"{base_path}/model_{i+1}_seed{seed}/best_model.zip"
   if os.path.exists(best_model_path):
      best_model = DQN.load(best_model_path, env=eval_env)
      obs = eval_env.reset()
      done = [False]
      final_balance = 0
   else:
      print(f"No se encontró el mejor modelo para seed {seed}")
      results.append({
         "seed": seed,
         "final_balance": 0,
         "drawdown": 0,
         "win_trades": 0,
         "lose_trades": 0,
         "winrate": 0,
         "model_path": "No encontrado"
      })

      while not done[0]:
        action, _ = best_model.predict(obs, deterministic= True)
        obs, reward, done, info = eval_env.step(action)
        final_balance = info[0]['balance']

      
      # ENTORNO DE EVALUACIÓN CON MONITOR #
      try:
        raw_eval_env = eval_env.envs[0].env  # Si hay Monitor
      except AttributeError:
        raw_eval_env = eval_env.envs[0]  # Si no hay Monitor
      drawdown = raw_eval_env.max_drawdown
      trades = raw_eval_env.trades
      closed_trades = [t for t in trades if t["type"] in ["close", "forced_close"]]
      winning_trades = [t for t in closed_trades if t.get("profit", 0) > 0]
      losing_trades = [t for t in closed_trades if t.get("profit", 0) <= 0]
      
      
      winrate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0
      results.append({
        "seed": seed,
        "final_balance" :  final_balance,
        "drawdown": drawdown,
        "win_trades": len(winning_trades),
        "lose_trades": len(losing_trades),
        "winrate": winrate,
        "model_path": best_model_path
      })

   ###########################
   ####### BACKTESTING #######
   ###########################
   print(f"Iniciando testing en 3..2..1..")
   time.sleep(3)
   obs = env.reset()
   done = [False]
   balances = []
   drawdowns = []
   tipos = []
   trades = []
 
   while not done[0]:
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      balances.append(info[0]['balance'])
      drawdowns.append(info[0]['drawdown'])
      tipos.append(info[0]['trade_type'])
 
      trades.append(f"Step: {info[0]['step']}, Tipo_trade: {info[0]['trade_type']}, Action: {action}, Result: {info[0]['result']} Reward: {reward[0]:.2f}, Balance: {info[0]['balance']:.2f}, Drawdown: {info[0]['drawdown']:.2f}")
      print(f"Step: {info[0]['step']}, Tipo_trade: {info[0]['trade_type']}, Action: {action}, Result: {info[0]['result']} Reward: {reward[0]:.2f}, Balance: {info[0]['balance']:.2f}, Drawdown: {info[0]['drawdown']:.2f}")
 
      
 
   output_resultados = "multiple_ml_learning"
   os.makedirs(output_resultados, exist_ok=True)
 
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   filename = f"ml_seed_{seed}_steps_{steps}_date_{timestamp}_bal_{balances[-1]:.2f}_dwmax_{max(drawdowns)}_maxbal_{max(balances):.2f}_minbal_{min(balances):.2f}"
 
   with open(os.path.join(output_resultados, filename), 'w') as f:
     f.write("### RESULTADOS ###\n")
     f.write(f"SEED: {seed}")
     f.write(f"\nBalance final: {balances[-1]:.2f}\n")
     f.write(f"Drawdown máximo: {max(drawdowns):.2f}\n")
     f.write(f"Balance máximo alcanzado: {max(balances):.2f}\n")
     f.write(f"Balance mínimo alcanzado: {min(balances):.2f}\n")
     f.write("\n")
      
     ##Insertar el contenido de TradingEnv.py ##
     f.write("\n\n")
     with open("TradingEnv.py", 'r') as envfile:
       f.write(envfile.read())

     ##Guardar los detalles de cada trade
     f.write("\n\nTRADES:\n")
     f.write("\n".join(trades))
   ### CALCULAR EFICIENCIA DE LA ESTRATEGIA ###
   print(f"\nBalance final: {balances[-1]:.2f}")
   print(f"Drawdown máximo: {max(drawdowns):.2f}")
   print(f"Balance máximo alcanzado: {max(balances):.2f}")
   print(f"Balance mínimo alcanzado: {min(balances):.2f}")




results.sort(key=lambda x: x['final_balance'], reverse = True)
for r in results:
    print(f"Seed: {r['seed']} | Balance Final: {r['final_balance']:.2f} | Winrate: {r['winrate']:.2f}% | Drawdown: {r['drawdown']:.2f} | Modelo: {r['model_path']}")

# with open(os.path.join(output_resultados, filename), 'w') as f:
   
#    f.write("RESULTADOS\n")
#    f.write(f"\nBalance final: {balances[-1]:.2f}\n")
#    f.write(f"Drawdown máximo: {max(drawdowns):.2f}\n")
#    f.write(f"Balance máximo alcanzado: {max(balances):.2f}\n")
#    f.write(f"Balance mínimo alcanzado: {min(balances):.2f}\n")
#    f.write("\n")

#    ## Insertar el contenido de TradingEnv.py ##
#    f.write("CONTENIDO DE TradingEnv.py --- \n\n")
#    with open("TradingEnv.py", "r") as env_file:
#       f.write(env_file.read())
   
#    ##Guardar los detalles de cada trade
#    f.write("\n\nTRADES:\n")
#    f.write('\n'.join(trades))

# raw_env = env.envs[0] #Accede al entorno original dentro del DummyVecEnv
# trades = raw_env.trades

try:
  env.close()
except:
    pass

try:
  eval_env.close()
except:
   pass
