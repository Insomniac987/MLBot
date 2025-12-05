import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TradingEnv import TradingEnv
from tqdm import tqdm
from stable_baselines3 import DQN


# ================= CONFIGURACIÓN =======================
MODELOS = [
    # "expert_professional_bots/agressive_winner_4M.zip",
    "expert_professional_bots/super_winner_8M_best_model.zip",
    "expert_professional_bots/super_winner_8M.zip",
    "expert_professional_bots/coke_SHORTER_8M_best.zip",
    "expert_professional_bots/coke_SHORTER_8M.zip"
    # "expert_professional_bots/smart_holder_2M.zip",
    # "expert_professional_bots/lucky_7M_best_model.zip",
    # "expert_professional_bots/lucky_7M.zip"
]
N_EPISODIOS = 100             # cuántas veces probar cada modelo
DATOS_PATH = "Data.csv"
WINDOW_SIZE = 50
PLOT_PATH = "equity_comparison.png"
# ========================================================


def evaluar_modelo(model_path, env, episodios=3):
    """Evalúa un modelo DQN durante N episodios."""
    print(f"\n🚀 Evaluando modelo: {model_path}")
    model = DQN.load(model_path, env=env, device="cpu")
    episodios_data = []
    all_equities = []

    for ep in tqdm(range(episodios)):
        obs, _ = env.reset()
        done = False
        rewards, balances, trades = [], [], 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            rewards.append(reward)
            balances.append(env.balance)
            if info.get("trade_executed"):
                trades += 1

        episodios_data.append({
            "episode": ep + 1,
            "final_balance": env.balance,
            "max_balance": max(balances),
            "min_balance": min(balances),
            "max_drawdown": 1 - (min(balances) / max(balances)),
            "mean_reward": np.mean(rewards),
            "trades": trades,
            "winrate": info.get("winrate", 0)
        })

        all_equities.append(balances)

    df = pd.DataFrame(episodios_data)
    resumen = df.mean().to_dict()
    resumen["model"] = os.path.basename(model_path)
    resumen["equities"] = all_equities
    return resumen, df


def calcular_score(row):
    """
    Calcula un puntaje ponderado para cada modelo:
    - 60% Balance final
    - 20% Winrate
    - 20% (1 - Max Drawdown)
    """
    return (
        0.6 * row["final_balance_norm"] +
        0.2 * row["winrate_norm"] +
        0.2 * (1 - row["max_drawdown_norm"])
    )


def main():
    print("📊 COMPARADOR DE MODELOS DQN\n")

    #Cargar datos
    df = pd.read_csv("Data.csv")
    df = df.rename(
        columns={
            'Precio de Cierre': 'close',
            'Volumen':'volume'
            }
    )

    env = TradingEnv(df, window_size=WINDOW_SIZE)
    resultados = []

    for model_path in MODELOS:
        if not os.path.exists(model_path):
            print(f"⚠️ No se encontró {model_path}, se omite.")
            continue

        resumen, df_detalle = evaluar_modelo(model_path, env, N_EPISODIOS)
        resultados.append(resumen)
        df_detalle.to_csv(f"detalles_{os.path.basename(model_path)}.csv", index=False)

    if not resultados:
        print("❌ No se pudo evaluar ningún modelo.")
        return

    # ===== Tabla resumen =====
    df = pd.DataFrame(resultados)
    df = df[
        ["model", "final_balance", "max_drawdown", "mean_reward", "trades", "winrate"]
    ]

    # Normalizar métricas (para poder compararlas)
    df["final_balance_norm"] = (df["final_balance"] - df["final_balance"].min()) / (df["final_balance"].max() - df["final_balance"].min() + 1e-9)
    df["winrate_norm"] = (df["winrate"] - df["winrate"].min()) / (df["winrate"].max() - df["winrate"].min() + 1e-9)
    df["max_drawdown_norm"] = (df["max_drawdown"] - df["max_drawdown"].min()) / (df["max_drawdown"].max() - df["max_drawdown"].min() + 1e-9)

    df["score"] = df.apply(calcular_score, axis=1)
    df = df.sort_values(by="score", ascending=False)

    # Mostrar resumen
    print("\n🏁 RESULTADOS COMPARATIVOS:")
    print(df[["model", "final_balance", "max_drawdown", "winrate", "score"]].to_string(index=False))

    df.to_csv("comparacion_modelos.csv", index=False)
    print("\n📁 Resultados guardados en 'comparacion_modelos.csv'")

    # ===== Determinar mejor modelo =====
    mejor = df.iloc[0]
    print(f"\n🏆 MEJOR MODELO SEGÚN SCORE: {mejor['model']}")
    print(f"   🔹 Score: {mejor['score']:.3f}")
    print(f"   💰 Balance Final: {mejor['final_balance']:.2f}")
    print(f"   📈 Winrate: {mejor['winrate']:.2f}")
    print(f"   📉 Drawdown: {mejor['max_drawdown']:.3f}")

    # ===== Gráfico de Equity Curves =====
    plt.figure(figsize=(10, 6))
    for res in resultados:
        equities = np.mean(
            [np.interp(np.linspace(0, 1, 300), np.linspace(0, 1, len(eq)), eq)
             for eq in res["equities"]], axis=0)
        plt.plot(equities, label=res["model"])

    plt.title("📈 Comparación de Curvas de Balance (Equity Curve)")
    plt.xlabel("Progreso del episodio (%)")
    plt.ylabel("Balance promedio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()

    print(f"\n🖼️ Gráfico guardado como: {PLOT_PATH}")


if __name__ == "__main__":
    main()
