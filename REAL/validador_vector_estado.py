import numpy as np
from stable_baselines3 import DQN
import inspect
import pandas as pd

# ==== 1. Cargar modelo entrenado ====
model = DQN.load("../BACKTESTING/expert_professional_bots/super_winner_8M_best_model")
expected_shape = model.observation_space.shape[0]
print("📐 El modelo espera un vector de dimensión:", expected_shape)

# ==== 2. Importar tu get_state desde bot_real_v2 ====
from bot_real import get_state, WINDOW_SIZE

# ==== 3. Crear DF dummy con todas las columnas necesarias ====
cols = ["close","RSI_14","ADX_14","OBV","MACD_hist",
        "CMF_20","StochRSI_K","BB_Width_20"]

df_dummy = pd.DataFrame(
    np.random.random((WINDOW_SIZE, len(cols))),
    columns=cols
)

# ==== 4. Generar estado dummy ====
cur_pct = 0.01
equity_change = 0.01
trade_duration = 5
drawdown = -0.02
pos_vector = [1,0,0]  # simulando LONG

state = get_state(df_dummy, cur_pct, equity_change, trade_duration, drawdown, pos_vector)

vector_length = state.shape[1]
print("🧪 Vector generado por get_state:", vector_length)

# ==== 5. Validación final ====
if vector_length == expected_shape:
    print("✅ PERFECTO: Las dimensiones COINCIDEN EXACTAMENTE.")
else:
    print("❌ ERROR: Mismatch de dimensiones")
    print("El modelo espera:", expected_shape)
    print("Tu bot produce:", vector_length)