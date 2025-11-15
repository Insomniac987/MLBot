import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import datetime
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

with open("config_dev.json", "r") as f:
    config = json.load(f)

# Config globals
COMISION = config.get("strategy", {}).get("comision", 0.0005)
SLIPPAGE = config.get("strategy", {}).get("slippage", 0.0005)
FACTOR_SEGURIDAD = config.get("strategy", {}).get("factor_seguridad", 0.25)

apalancamiento = config.get("common", {}).get("apalancamiento")
initial_balance = config.get("strategy", {}).get("saldo_inicial", 200)
max_drawdown = config.get("strategy", {}).get("max_drawdown", 0.75)
window_size = config.get("datos", {}).get("window_size", 50)
episode_steps_cfg = config.get("datos", {}).get("episode_steps", 10000)
timestamp = datetime.datetime.now().strftime("%Y%md_%H%M%S")

# map leverage string to numeric
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
        leverage = 1


class TradingEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window_size: int = window_size, initial_balance: float = initial_balance, leverage: int = leverage, modo_backtest: bool = False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.modo_backtest = modo_backtest
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.leverage = int(leverage)

        # required columns
        self.feature_cols = ['close', 'RSI_14', 'ADX_14', 'OBV', 'MACD_hist', 'CMF_20', 'StochRSI_K', 'BB_Width_20']
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise AssertionError(f"Faltan columnas: {missing}")

        # quick dataset length check
        if len(self.df) <= self.window_size + episode_steps_cfg:
            raise ValueError(f"Dataset demasiado pequeño: {len(self.df)} filas, se requieren al menos {self.window_size + episode_steps_cfg + 1}")

        # action space (DQN friendly)
        # 0 = Hold, 1 = Long Buy, 2 = Long Close, 3 = Short Sell, 4 = Short Close
        self.action_space = spaces.Discrete(5)

        # observation: flattened window + state vector
        hist_feats = self.window_size * len(self.feature_cols)
        state_feats = 8  # [balance_norm, current_profit, equity_change, trade_duration, drawdown, pos_onehot(3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(hist_feats + state_feats,), dtype=np.float32)

        # episode settings
        self.episode_steps_cfg = episode_steps_cfg

        # init state
        self.reset()

    def reset(self, seed=None, options=None, episode_steps: int = None):
        super().reset(seed=seed)
        # episode length
        self.episode_steps = self.episode_steps_cfg if episode_steps is None else episode_steps

        # choose start & end
        max_start = len(self.df) - self.episode_steps - 1
        if max_start <= self.window_size:
            raise ValueError("Dataset demasiado pequeño para el episodio configurado")

        if self.modo_backtest:
            self.start_step = self.window_size
            self.end_step = len(self.df) - 1
            self.episode_steps = self.end_step - self.start_step
        else:
            self.start_step = np.random.randint(self.window_size, max_start)
            self.end_step = self.start_step + self.episode_steps

        self.current_step = self.start_step

        # account state
        self.balance = float(self.initial_balance)
        self.prev_balance = float(self.initial_balance)
        self.position = 0  # 1 long, -1 short, 0 neutral
        self.entry_price = 0.0
        self.peak_balance = float(self.initial_balance)
        self.max_drawdown = 0.0
        self.trades = []
        self.episode_lines = []
        self.prev_max_dd = 0
        self.hold_inpos_steps = 0
        self.hold_outpos_steps = 0
        self.noop_steps = 0
        self.prev_max_dd = 0.0
        self.stats = {"ganadoras": 0, "perdedoras": 0, "forzadas": 0}

        # capital per trade variable (set on open)
        self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD

        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        obs_raw = window[self.feature_cols].values
        # normalize per-window
        means = obs_raw.mean(axis=0)
        stds = obs_raw.std(axis=0) + 1e-8
        obs = ((obs_raw - means) / stds).flatten()

        # position one-hot
        if self.position == 1:
            pos = [1, 0, 0]
        elif self.position == -1:
            pos = [0, 1, 0]
        else:
            pos = [0, 0, 1]

        balance_norm = self.balance / max(self.initial_balance, 1)

        # current unrealized profit (pct)
        if self.position != 0:
            current_price = self.df.iloc[self.current_step]['close']
            cur_pct = (current_price - self.entry_price) / max(self.entry_price, 1e-8)
            if self.position == -1:
                cur_pct = -cur_pct
        else:
            cur_pct = 0.0

        if self.position != 0 and self.trades:
            trade_duration = (self.current_step - self.trades[-1]['step']) / max(self.episode_steps, 1)
        else:
            trade_duration = 0.0

        equity_change = (self.balance - self.prev_balance) / max(self.initial_balance, 1)
        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)

        state_vector = np.array([balance_norm, cur_pct, equity_change, trade_duration, drawdown] + pos, dtype=np.float32)
        full_obs = np.concatenate([obs.astype(np.float32), state_vector])
        return full_obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        end_reason = None

        price = self.df.iloc[self.current_step]['close']
        pct_stop = config.get("strategy", {}).get("stop_loss", 0.05)

        closed_this_step = False
        last_close_profit = 0.0
        last_close_pct = 0.0
        current_profit_pct = 0.0
        prev_profit_pct = 0.0
        step_trade_type = None

        # detect valid actions (simple masking behaviour)
        has_long = self.position == 1
        has_short = self.position == -1
        if not has_long and not has_short:
            valid_actions = [1, 3]
        elif has_long:
            valid_actions = [0, 2]
        else:
            valid_actions = [0, 4]

        if action not in valid_actions:
            # don't change position, small noop penalty (helps DQN learn masking implicitly)
            step_trade_type = "noop"
            self.noop_steps += 1
        else:
            # execute selected valid action
            if action == 0:
                step_trade_type = 'hold_inpos' if self.position != 0 else 'hold_outpos'
            elif action == 1 and self.position == 0:
                # open long
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD
                notional = self.capital_usado_actual * self.leverage
                comision_apertura = notional * COMISION
                self.balance -= comision_apertura
                self.entry_price = price * (1 + SLIPPAGE)
                self.position = 1
                self.trades.append({'type': 'LONG BUY', 'step': self.current_step, 'entry_price': self.entry_price})
                step_trade_type = 'LONG BUY'
            elif action == 2 and self.position == 1:
                # close long
                capital_usado = self.capital_usado_actual
                notional = capital_usado * self.leverage
                comision_cierre = notional * COMISION
                exit_price = price * (1 - SLIPPAGE)
                pct_change = (exit_price - self.entry_price) / max(self.entry_price, 1e-8)
                impacto = pct_change * self.leverage
                profit = capital_usado * impacto
                loss_limit = -pct_stop * capital_usado
                if profit < loss_limit:
                    profit = loss_limit
                self.balance += profit - comision_cierre
                self.peak_balance = max(self.peak_balance, self.balance)
                drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
                self.max_drawdown = max(self.max_drawdown, drawdown)

                if profit > 0:
                    self.stats['ganadoras'] += 1
                else:
                    self.stats['perdedoras'] += 1

                self.trades.append({'type': 'LONG CLOSE', 'step': self.current_step, 'exit_price': exit_price, 'position': 'long', 'profit': profit})
                step_trade_type = 'LONG CLOSE'
                closed_this_step = True
                last_close_profit = profit
                last_close_pct = pct_change
                self.position = 0
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD
            elif action == 3 and self.position == 0:
                # open short
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD
                notional = self.capital_usado_actual * self.leverage
                comision_apertura = notional * COMISION
                self.balance -= comision_apertura
                self.entry_price = price * (1 - SLIPPAGE)
                self.position = -1
                self.trades.append({'type': 'SHORT SELL', 'step': self.current_step, 'entry_price': self.entry_price})
                step_trade_type = 'SHORT SELL'
            elif action == 4 and self.position == -1:
                # close short
                capital_usado = self.capital_usado_actual
                notional = capital_usado * self.leverage
                comision_cierre = notional * COMISION
                exit_price = price * (1 + SLIPPAGE)
                pct_change = (self.entry_price - exit_price) / max(self.entry_price, 1e-8)
                impacto = pct_change * self.leverage
                profit = capital_usado * impacto
                loss_limit = -pct_stop * capital_usado
                if profit < loss_limit:
                    profit = loss_limit
                self.balance += profit - comision_cierre
                self.peak_balance = max(self.peak_balance, self.balance)
                drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
                self.max_drawdown = max(self.max_drawdown, drawdown)

                if profit > 0:
                    self.stats['ganadoras'] += 1
                else:
                    self.stats['perdedoras'] += 1

                self.trades.append({'type': 'SHORT CLOSE', 'step': self.current_step, 'exit_price': exit_price, 'position': 'short', 'profit': profit})
                step_trade_type = 'SHORT CLOSE'
                closed_this_step = True
                last_close_profit = profit
                last_close_pct = pct_change
                self.position = 0
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD

        # advance step index
        self.current_step += 1

        # count hold/noop
        if step_trade_type == 'noop':
            self.noop_steps += 1
        elif step_trade_type == 'hold_inpos':
            self.hold_inpos_steps += 1
        elif step_trade_type == 'hold_outpos':
            self.hold_outpos_steps += 1

        # termination conditions
        current_dd = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
        if current_dd > max_drawdown:
            terminated = True
            end_reason = 'liquidation'

        if self.current_step >= self.end_step:
            terminated = True
            end_reason = end_reason or 'end_of_data'

        if (not self.modo_backtest) and ((self.current_step - self.start_step) >= self.episode_steps):
            truncated = True
            end_reason = end_reason or 'time_limit'

        # forced close if episode ends and position open
        if (terminated or truncated) and self.position != 0:
            final_price = price
            capital_usado = getattr(self, 'capital_usado_actual', self.balance * FACTOR_SEGURIDAD)
            notional = capital_usado * self.leverage
            comision = notional * COMISION
            exit_price = (final_price * (1 - SLIPPAGE)) if self.position == 1 else (final_price * (1 + SLIPPAGE))
            pct_change = (exit_price - self.entry_price) / max(self.entry_price, 1e-8)
            if self.position == -1:
                pct_change = -pct_change
            impacto = pct_change * self.leverage
            profit = capital_usado * impacto
            loss_limit = -pct_stop * capital_usado
            if profit < loss_limit:
                profit = loss_limit
            self.balance += profit - comision
            self.peak_balance = max(self.peak_balance, self.balance)
            drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
            self.max_drawdown = max(self.max_drawdown, drawdown)
            self.trades.append({'type': 'FORCED_CLOSE', 'step': self.current_step, 'exit_price': exit_price, 'position': 'long' if self.position == 1 else 'short', 'profit': profit})
            if profit > 0:
                self.stats['ganadoras'] += 1
            else:
                self.stats['perdedoras'] += 1
            self.stats['forzadas'] += 1
            closed_this_step = True
            last_close_profit = profit
            last_close_pct = pct_change
            self.position = 0

        #################################
        #### REWARDS Y PENALIZACIONES####
        #################################

        # 1. Inicializar variables base
        profit_reward = 0
        pnl_reward = 0
        noop_reward = 0
        hold_in_reward = 0
        hold_out_reward = 0
        dd_reward = 0
        ddrec_reward = 0

        # 2. Cargar los factores desde config.json
        factores = config.get("factores_rewards", {})
        f_pft = factores.get("pft", {})
        f_pnl = factores.get("pnl", {})
        f_hold_in = factores.get("hold_in", {})
        f_hold_out = factores.get("hold_out", {})
        f_noop = factores.get("noop", {})
        f_dd = factores.get("dd", {})
        f_ddrec = factores.get("ddrec", {})

        # 3. Definir recompensas base según tipo de acción
        if closed_this_step:
            # --- A. Recompensa por cerrar operación ---
            if last_close_profit > 0:
                profit_reward = 1.0   # Ganancia
            else:
                profit_reward = -1.0  # Pérdida

        elif step_trade_type == 'noop':
            # --- B. Penalización por acción inválida ---
            noop_reward = -1.0

        elif step_trade_type == 'hold_inpos':
            # --- C. Mantener posición abierta ---
            hold_in_reward = -1.0  # Pequeña penalización para no quedarse eternamente

            # Calcular PnL no realizado como pista
            current_price = self.df.iloc[self.current_step]['close']
            current_profit_pct = (current_price - self.entry_price) / max(self.entry_price, 1e-8)
            if self.position == -1:  # Si es short, invierte el signo
                current_profit_pct = -current_profit_pct
            pnl_reward = np.tanh((current_profit_pct- prev_profit_pct)/5)
            prev_profit_pct = current_profit_pct

        elif step_trade_type == 'hold_outpos':
            # --- D. Esperar sin posición ---
            hold_out_reward = -1.0

        # DRAWDOWN

        # Penalización incremental por drawdown, pero mucho más suave y capped
        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
        worse_drawdown = max(drawdown - self.prev_max_dd, 0.0)
        penalty = worse_drawdown * 20
        dd_reward = np.clip(penalty, -1, 1.0)

        # Reward por recuperación de un drawdown
        if drawdown < self.prev_max_dd:
            recovery = (self.prev_max_dd - drawdown) * 1
            ddrec_reward += np.clip(recovery, 0, 1)

        self.prev_max_dd = max(self.prev_max_dd, drawdown)

        # 4. Calcular recompensa total ponderada
        total_reward = (
            (profit_reward * f_pft) +
            (pnl_reward * f_pnl) +
            (noop_reward * f_noop) +
            (hold_in_reward * f_hold_in) +
            (hold_out_reward * f_hold_out) +
            (dd_reward * f_dd) +
            (ddrec_reward * f_ddrec)
        )

        # 5. Normalizar rango
        reward = np.tanh(total_reward/5)

        # 6. Actualizar trackers
        self.prev_balance = self.balance
        self.prev_action = int(action)

        # episode-level logging when episode ends
        info = {
            'trade_type': step_trade_type,
            'action': int(action),
            'balance': self.balance,
            'drawdown': self.max_drawdown,
            'profit': last_close_profit if closed_this_step else 0.0,
            'step': self.current_step,
            'HOLDS_INPOS': int(self.hold_inpos_steps),
            'HOLDS_OUTPOS': int(self.hold_outpos_steps),
            'NOOPS': int(self.noop_steps),
        }

        if end_reason:
            info['end_reason'] = end_reason

        if terminated or truncated:
            info['episode_trades'] = list(self.trades)
            line = {
            f"EPISODE END | Balance: {self.balance:.2f}"
            f"| Max DD: {self.max_drawdown:.3f}"
            f"| Trades: {len(self.trades)}"
            f"| Wins: {self.stats['ganadoras']}"
            f"| Losses: {self.stats['perdedoras']}"
            f"| Effectiveness (winrate): {(self.stats['ganadoras']/self.stats['perdedoras'] if self.stats != 0 else 0):0.2f}%"
        }
            self.episode_lines.append(line)

            # Save accumulated file
            newlogs = "newlogs"
            with open(os.path.join(newlogs, f"{timestamp}_learning.csv"), "a", buffering=1) as f:
                f.write(f"{line}\n")
        
        return self._get_observation(), reward, terminated, truncated, info