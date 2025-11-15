import gymnasium as gym
import warnings
from gymnasium import spaces
import numpy as np
import pandas as pd
import json

### NO MOSTRAR RUNTIME WARNINGS (ERROR DE REWARD MUY GRANDE) ###
warnings.filterwarnings("ignore", category=RuntimeWarning)

## IMPORTAR CONFIGURACIÓN ##

with(open("config_dev.json", "r")) as f:
   config = json.load(f)

### INICIALIZAR VARIABLES ###

COMISION = config.get("strategy", {}).get("comision") # 0.05% taker fee en Binance Futures
SLIPPAGE = config.get("strategy", {}).get("slippage") # 0.05% slippage promedio por orden de mercado
FACTOR_SEGURIDAD = config.get("strategy", {}).get("factor_seguridad") # Igual que en el bot real, utilizar un N% del capital en lugar de un 100%

apalancamiento = config.get("common", {}).get("apalancamiento")
initial_balance = config.get("strategy", {}).get("saldo_inicial")
max_profit_loss = config.get("strategy", {}).get("max_profit_loss")
threshold_ganancia = config.get("strategy", {}).get("threshold_ganancia")
max_drawdown = config.get("strategy", {}).get("max_drawdown")
window_size = config.get("datos", {}).get("window_size")
episode_steps = config.get("datos", {}).get("episode_steps")


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
      print("Opción no válida, usando 1x")

print(f"El apalancamiento ha sido seteado a {leverage}")

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=window_size, initial_balance=initial_balance, leverage=leverage, modo_backtest=False):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.modo_backtest = modo_backtest
        self.window_size = window_size
        ### Inicializar balance ###
        self.initial_balance = initial_balance
        ### Inicializar leverage ###
        self.leverage = leverage
        self.max_trade_profit = 0
        self.max_trade_profit_lev = 0
        self.stats = {"ganadoras": 0, "perdedoras": 0, "forzadas": 0}

        # Usar episode_steps del config (no self.episode_steps)
        episode_steps_config = config.get("datos", {}).get("episode_steps")

        #Verificar que el dataframe contiene las columnas necesarias
        required_columns = ['close', 'RSI_14', 'ADX_14', 'OBV', 'MACD_hist', 'CMF_20', 'StochRSI_K', 'BB_Width_20']
        assert all(col in self.df.columns for col in required_columns), f"Faltan columnas: {[col for col in required_columns if col not in self.df.columns]}"

        # Verificar que el dataset es suficientemente grande
        assert len(self.df) > self.window_size + episode_steps_config, f"Dataset demasiado pequeño: {len(self.df)} filas, pero se necesitan al menos {self.window_size + episode_steps_config + 1}"

        # Precalcular medias y desviaciones estándar para normalización eficiente
        self.feature_cols = ['close', 'RSI_14', 'ADX_14', 'OBV', 'MACD_hist', 'CMF_20', 'StochRSI_K', 'BB_Width_20']
        
        self.feature_means = self.df[self.feature_cols].mean().values
        self.feature_stds = self.df[self.feature_cols].std().values + 1e-8 #(evita difisión por cero)

        #Nuevo espacio de observación: datos históricos + estado actual
        historical_features = self.window_size * len(self.feature_cols) # 7 features por vela
        state_features = 8 #balance_norm, current_profit, equity_change, trade_duration, drawdown, one-hot pos (3)

        self.action_space = spaces.Discrete(5)  # 0=Hold, 1=Long Buy, 2=Long Close, 3=Short Sell, 4=Short Close
        ### Definir el espacio de observación del entorno Gymnasium ###
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            ### El número de features por vela --> window_size (7) ###
            # shape=(window_size, 7),
            shape =(historical_features + state_features,),
            dtype=np.float32
        )
        self.reset()

    ### Resetear el entorno una vez que terminó el episodio ###
    def reset(self, seed=None, options=None, episode_steps=episode_steps):
        super().reset(seed=seed)

        #Usar episode_steps del config
        episode_steps_config = config.get("datos", {}).get("episode_steps")
        
        # Longitud del episodio
        self.episode_steps = episode_steps_config

        # Elegir un índice inicial aleatorio
        max_start = len(self.df) - self.episode_steps - 1
        if max_start <= self.window_size:
            raise ValueError("El dataset es demasiado pequeño para el episode_steps configurado")
        
        if self.modo_backtest:
            # Un solo episodio que recorre hasta el final de los datos
            self.start_step = self.window_size
            self.end_step = len(self.df) - 1
            self.episode_steps = self.end_step - self.start_step
        else:
            self.start_step =np.random.randint(self.window_size, max_start)
            self.end_step = self.start_step + self.episode_steps

        self.current_step = self.start_step

         # Reset de variables de la cuenta
        self.balance = self.initial_balance
        self.prev_balance = self.initial_balance
        self.position = 0  # 1 = long, -1 = short, 0 = neutral
        self.entry_price = 0
        self.max_drawdown = 0
        self.peak_balance = self.initial_balance
        self.trades = []
        self.hold_inpos_steps = 0
        self.hold_outpos_steps = 0
        self.noop_steps = 0
        self.prev_max_dd = (self.peak_balance - self.balance) / max(self.peak_balance, 1)

        # Resetear stats por episodio y acumulador de DD incremental
        self.stats = {"ganadoras": 0, "perdedoras": 0, "forzadas": 0}
        return self._get_observation(), {}

    def _get_observation(self):
        ### extraer una ventana de datos ###
        # toma las últimas window_size velas(ej. 50,100...) hasta el current_step
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        ### seleccionar las features relevantes ###
        obs_raw = window[self.feature_cols].values
        # normalizar por mean/std de la ventana (no del dataset entero)
        means = obs_raw.mean(axis=0)
        stds  = obs_raw.std(axis=0) + 1e-8
        obs = (obs_raw - means) / stds
        obs = obs.flatten()

        ### Añadir estado actual ###
        # Codificar posicion one-hot
        if self.position == 1:
            position_encoded = [1,0,0] # Long
        elif self.position == -1:
            position_encoded = [0,1,0] # Short
        else:
            position_encoded = [0,0,1] # Neutral
        
        #Balance normalizado
        balance_normalized = self.balance / self.initial_balance

        # Profit actual si hay posición abierta
        if self.position != 0:
            current_price = self.df.iloc[self.current_step]['close']
            current_profit = (current_price - self.entry_price) / self.entry_price
            if self.position == -1:
                current_profit = -current_profit
        else:
            current_profit = 0.0
        
        # Duración del trade actual (normalizada para observación)
        if self.position != 0 and self.trades:
            trade_duration = (self.current_step - self.trades[-1]['step']) / self.episode_steps
        else:
            trade_duration = 0.0

        # Variación reciente del equity ('ultimo step)
        equity_change = (self.balance - self.prev_balance) / self.initial_balance

        #Drawdown actual (relativo al pico de balance)

        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
        
        # Crear vector de estado
        state_vector = [balance_normalized, current_profit, equity_change, trade_duration, drawdown] + position_encoded
        state_vector = np.array(state_vector, dtype=np.float32)

        #Combinar con datos históricos
        full_obs = np.concatenate([obs, state_vector])
        return full_obs.astype(np.float32)
    
    
    ###Se ejecuta una vez por cada caso ###

    ### UN STEP EQUIVALE A UNA FILA EN EL DATAFRAME ###

    def step(self, action):
        reward = 0
        info = {}
        truncated = False
        terminated = False
        end_reason = None
        trade = False
        profit = 0
        price = self.df.iloc[self.current_step]['close']
        pct_change = 0
        pct_stop = config.get("strategy", {}).get("stop_loss")

        # Inicializar flags de cierre
        closed_this_step = False
        last_close_pct = 0.0
        last_close_profit = 0.0
        step_trade_type = None
        
        ### FILTRO DE ACCIONES INVÁLIDAS (evita No-ops innecesarios) ###
        # Detectar si hay una posición activa
        has_long = self.position == 1
        has_short = self.position == -1
        
        # Acciones válidas seguón el contexto actual 
        if not has_long and not has_short:
            valid_actions = [1, 3] # Puede abrir long o short
        elif has_long:
            valid_actions = [0, 2] # Puede hacer hold o cerrar long
        elif has_short:
            valid_actions = [0,4] # Puede hacer hold o cerrar short
        else:
            valid_actions = [0] # Fallback seguro
        
        if action not in valid_actions:
            # Pequeña penalización, sin forzar HOLD ni alterar la posición
            step_trade_type = "noop"

        # Acción 0: HOLD
        if action == 0:
            if self.position == 0:
                step_trade_type = 'hold_outpos'
            else:
                step_trade_type = 'hold_inpos'
            pass

        # Acción 1: LONG
        #0=Hold, 1=Long Buy, 2=Long Close, 3=Short Sell 4= Short close
        elif action == 1:

            ###################
            #### LONG BUY  ####
            ###################

            if self.position == 0:
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD
                notional = self.capital_usado_actual * self.leverage
                comision_apertura = notional * COMISION
                self.balance -= comision_apertura
                self.max_trade_profit = 0
                self.max_trade_profit_lev = 0
                self.entry_price = price * (1 + SLIPPAGE)
                self.position = 1
                self.trades.append({
                    'type': 'LONG BUY',
                    'step': self.current_step,
                    'entry_price': self.entry_price
                })
                step_trade_type = 'LONG BUY'
                trade = True
            
        elif action == 2:

            ##################
            ### LONG CLOSE ###
            ##################
            if self.position == 1:

                ### Calular variables de notional y capital usado ###
                capital_usado = self.capital_usado_actual
                notional = capital_usado * self.leverage
                comision_cierre = notional * COMISION

                # Solamente en el cierre del long se resta el slippage
                exit_price = price * (1- SLIPPAGE)
                pct_change = (exit_price - self.entry_price) / self.entry_price
                # el impacto representa el porcentaje de ganancia o pérdida ajustado por el apalancamiento
                impacto = pct_change * self.leverage
                # la ganancia / pérdida real del trade
                profit = capital_usado * impacto
                loss_limit = -pct_stop * capital_usado                

                ##################################################
                ## 🔴 Stop Loss real (ej. 5% del capital usado) ##
                ##################################################

                if profit < loss_limit:
                   profit = loss_limit
                
                # actualizar el balance sumando el profit y restando comisión
                self.balance += profit - comision_cierre 

                ## CALCULAR DRAWDOWN ##
                self.peak_balance = max(self.peak_balance, self.balance)
                # Drawdown en decimales
                drawdown = (self.peak_balance - self.balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, drawdown)
                
                #######################
                ### REGISTRAR TRADE ###
                #######################

                # Estadísticas resultados (después del SL)
                if profit > 0:
                    self.stats["ganadoras"] += 1
                else:
                    self.stats["perdedoras"] += 1

                self.trades.append({
                    'type': 'LONG CLOSE',
                    'step': self.current_step,
                    'exit_price': exit_price,
                    'position': 'long',
                    'profit': profit,
                    'result': info.get('result') or '\033[92mtake profit\033[0m' if profit > 0 else '\033[91mstop loss\033[0m'
                })
                step_trade_type = 'LONG CLOSE'

                # Marcar cierre y guardar métricas para reward
                closed_this_step = True
                self.max_trade_profit_lev = 0
                last_close_pct = pct_change
                last_close_profit = profit
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD

                # Resetear la posición a cero
                self.position = 0

        # Acción 3: 
        elif action == 3:
            #0=Hold, 1=Long Buy, 2=Long Close, 3=Short Sell 4= Short close
            ##################
            ### SHORT SELL ###
            ##################

            #Si no hay posición abiertam entonces abrir una
            if self.position == 0:
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD
                notional = self.capital_usado_actual * self.leverage
                comision_apertura = notional * COMISION
                self.balance -= comision_apertura
                self.max_trade_profit = 0
                self.max_trade_profit_lev = 0
                self.entry_price = price * (1 - SLIPPAGE)
                self.position = -1
                self.trades.append({
                    'type': 'SHORT SELL',
                    'step': self.current_step,
                    'entry_price': self.entry_price
                })
                step_trade_type = 'SHORT SELL'
                trade = True
        
        # Acción 4
        elif action == 4:
            ###################
            ### SHORT CLOSE ###
            ###################

            if self.position == -1:

                ### Calular variables de notional y capital usado ###
                capital_usado = self.capital_usado_actual
                notional = capital_usado * self.leverage
                comision_cierre = notional * COMISION
                loss_limit = -pct_stop * capital_usado

                # Solamente en el cierre del SHORT se resta el slippage
                exit_price = price * (1+ SLIPPAGE)
                pct_change = (self.entry_price - exit_price) / self.entry_price
                impacto = pct_change * self.leverage
                profit = capital_usado * impacto # la ganancia / pérdida real del trade          

                ##################################################
                ## 🔴 Stop Loss real (ej. 5% del capital usado) ##
                ##################################################

                if profit < loss_limit:
                    profit = loss_limit
                
                # actualizar el balance sumando el profit y restando comisión                
                self.balance += profit - comision_cierre
                self.peak_balance = max(self.peak_balance, self.balance)
                drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
                self.max_drawdown = max(self.max_drawdown, drawdown)

                #######################
                ### REGISTRAR TRADE ###
                #######################

                self.trades.append({
                    'type': 'SHORT CLOSE',
                    'step': self.current_step,
                    'exit_price': exit_price,
                    'position': 'short',
                    'profit': profit,
                    'result': info.get('result') or '\033[92mtake profit\033[0m' if profit > 0 else '\033[91mstop loss\033[0m'
                })
                step_trade_type = 'SHORT CLOSE'

                # Estadísticas resultados (después del SL)
                if profit > 0:
                    self.stats["ganadoras"] += 1
                else:
                    self.stats["perdedoras"] += 1

                # Marcar cierre y guardar métricas para reward
                closed_this_step = True
                self.max_trade_profit_lev = 0
                last_close_pct = pct_change
                last_close_profit = profit
                self.capital_usado_actual = self.balance * FACTOR_SEGURIDAD

                # Resetear la posición a cero
                self.position = 0
            
            else:
                step_trade_type = 'noop'
                noop_reason = 'no_short_to_close'

        # Fallback claro
        if step_trade_type is None:
            if action == 0:
                step_trade_type = 'hold'
            else:
                step_trade_type = 'noop'
                noop_reason = noop_reason or 'no_effect'
        
        #### TERMINAR SI LA CUENTA LLEGA A UN DRAWDOWN MÁXIMO ####
                

        ## SI EL BALANCE LLEGA UN DRAWDOWN MÁXIMO (EJEMPLO 70%)
        current_dd = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
        if current_dd > max_drawdown:
            #Terminar el episodio por liquidación
            terminated = True
            end_reason = 'liquidation'
            print(f"###############################################")
            print(f"##### LA CUENTA HA SIDO QUEMADA, OOOOPSSS #####")
            print(f"###############################################")

        # Avanzar al siguiente indice del dataframe
        self.current_step += 1

        # Incrementar contadores para los logs (esto no afecta la recompensa)
        if step_trade_type == "noop":
            self.noop_steps += 1
        elif step_trade_type == "hold_inpos":
            self.hold_inpos_steps += 1
        elif step_trade_type == "hold_outpos":
            self.hold_outpos_steps += 1

        # Fin de datos
        if self.current_step >= self.end_step:
            terminated = True
            end_reason = end_reason or 'end_of_data'
        
        #limite de pasos el episodio (solo en entrenamiento)
        if (not self.modo_backtest) and ((self.current_step- self.start_step) >= self.episode_steps):
            truncated = True
            end_reason = end_reason or 'time_limit'

        ####################
        ## CIERRE FORZADO ##
        ####################

        # Cierre forzado si termina
        if (terminated or truncated) and self.position != 0:
            
            # Usar el precio del último paso procesado (evita lookahead)
            final_price = price

            ### Calular variables de notional y capital usado ###
            capital_usado = getattr(self, "capital_usado_actual",self.balance * FACTOR_SEGURIDAD)
            notional = capital_usado * self.leverage
            comision = notional * COMISION
            loss_limit = -pct_stop * capital_usado
            
            # Solamente calcular el slippage si se cierra la posición en short o long
            exit_price = (final_price * (1 - SLIPPAGE)) if self.position == 1 else (final_price * (1 + SLIPPAGE))
            pct_change = (exit_price - self.entry_price) / self.entry_price

            #ganancia para long o para short
            if self.position == -1:
              pct_change *= -1

            impacto = pct_change * self.leverage
            profit = capital_usado * impacto
            
            # 🔴 Stop Loss real (ej. 5% del capital usado)
            loss_limit = -pct_stop * capital_usado
            if profit < loss_limit:
                profit = loss_limit           
            
            self.balance += profit - comision

            # Penalización por drawdown
            self.peak_balance = max(self.peak_balance, self.balance)
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)

            self.trades.append({
                'type': 'FORCED CLOSE',
                'step': self.current_step,
                'exit_price': exit_price,
                'position': 'long' if self.position == 1 else 'short',
                'profit': profit,
                'result': '\033[92mtake profit\033[0m' if profit > 0 else '\033[91mstop loss\033[0m'
            })
            step_trade_type = 'FORCED CLOSE'
            
            if profit > 0:
                self.stats["ganadoras"] += 1
            else:
                self.stats["perdedoras"] += 1
            self.stats["forzadas"] += 1

            # Marcar cierre y guardar métricas para reward
            closed_this_step = True
            last_close_pct = pct_change
            last_close_profit = profit
            self.position = 0

        #################################
        #### REWARDS Y PENALIZACIONES####
        #################################

        # Guardar el porcentaje de ganancia / perdida con leverage
        last_close_pct_lev = last_close_pct * self.leverage

        #Inicializar rewards a 0

        hold_inpos_reward = 0
        hold_outpos_reward= 0
        noop_reward= 0
        dd_recovery_reward= 0
        dd_penalty= 0
        hold_100_inpos_penalty= 0
        hold_50_outpos_penalty= 0
        small_win_reward= 0
        small_loss_penalty= 0
        big_win_reward= 0
        big_loss_penalty= 0
        profit_reward= 0
        loss_penalty= 0
        balance_reward=0
        open_trade_reward = 0

        ## Recompensa más suave y estable por % de cambio de balance
        balance_pct = (self.balance - self.prev_balance) / max(self.initial_balance, 1)
        balance_reward = np.log1p(abs(balance_pct)) * np.sign(balance_pct) * 100
        reward += balance_reward

        # Reward por cerrar en ganancia
        if last_close_profit > 0:
            profit_bonus = 1
            profit_reward = profit_bonus * (last_close_profit / self.initial_balance)
            reward += profit_reward
        # Penalizar por cerrar en pérdida
        elif last_close_profit <= 0:
            loss_punish = 0.5
            loss_penalty = loss_punish * abs(last_close_profit / self.initial_balance)
            reward -= loss_penalty
        
        ### REWARD / PENALIZACIÓN POR CALIDAD DE TRADES ###
        if last_close_profit > 0:
            # REWARD por trades con profits pequeños
            if closed_this_step and abs(last_close_pct_lev) < 0.03:
                small_win_reward = min(last_close_pct_lev, 0.04)
                reward += small_win_reward
            # REWARD por trades con profits grandes
            if closed_this_step and abs(last_close_pct_lev) > 0.05:
                big_win_reward = min(last_close_pct_lev, 0.8) * 10
                reward += big_win_reward
        elif last_close_profit <0:
            # PENALIZACIÓN pequeña por perdidas pequeñas
            if closed_this_step and abs(last_close_pct_lev) < 0.03:
                small_loss_penalty = min(last_close_pct_lev, 0.04)
                reward -= small_loss_penalty
            # PENALIZACIÓN grande por perdidas pequeñas
            if closed_this_step and abs(last_close_pct_lev) > 0.05:
                big_loss_penalty = min(last_close_pct_lev, 0.8)
                reward -= big_loss_penalty


        # Penalización por llevar mucho tiempo sin operar
        if getattr(self, "hold_inpos_steps", 0) > 100:
            hold_100_inpos_penalty = 5
            reward -= hold_100_inpos_penalty
        if getattr(self, "hold_outpos_steps", 0) > 50:
            hold_50_outpos_penalty = 5
            reward -= hold_50_outpos_penalty

        # Penalización incremental por drawdown, pero mucho más suave y capped
        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1)
        worse_drawdown = max(drawdown - self.prev_max_dd, 0.0)
        penalty = worse_drawdown * 100  # antes era *100 -> demasiado duro
        dd_penalty = np.clip(penalty, 0.0, 1.0)
        reward -= dd_penalty
        self.prev_max_dd = max(self.prev_max_dd, drawdown)

        # Reward por recuperación de un drawdown
        if drawdown < self.prev_max_dd:
            dd_recovery_reward = (self.prev_max_dd - drawdown) * 1
            reward += dd_recovery_reward

        #Penalizar hold
        if step_trade_type == 'hold_inpos':
            hold_inpos_reward = 0.1
            reward -= hold_inpos_reward
        elif step_trade_type == 'hold_outpos':
            hold_outpos_reward = 0.1
            reward -= hold_outpos_reward
        
        # Penalización Trades inválidos (noop)
        if step_trade_type == 'noop':
            noop_reward = 5
            reward -= noop_reward

        # Incentivar a que abra trades
        if trade == True:
            open_trade_reward = 1
            reward += open_trade_reward

        ### LIMITAR REWARDS ###

        reward = np.clip(reward, -50, 50)

        print(
        f"Hold_inpos_reward: {-hold_inpos_reward}\n"
        f"Hold_outpos_reward: {-hold_outpos_reward}\n"
        f"Noop_reward: {-noop_reward}\n"
        f"DD_Recovery_reward: {dd_recovery_reward}\n"
        f"DD_Penalty: {-dd_penalty}\n"
        f"hold_100_inpos_penalty: {-hold_100_inpos_penalty}\n"
        f"hold_50_outpos_penalty {-hold_50_outpos_penalty}\n"
        f"Small win reward: {small_win_reward}\n"
        f"Small loss penalty: {-small_loss_penalty}\n"
        f"Big win reward: {big_win_reward}\n"
        f"Big loss penalty: {-big_loss_penalty}\n"
        f"Profit reward {profit_reward}\n"
        f"Loss penalty: {-loss_penalty}\n"
        f"Balance_reward: {balance_reward}\n"
        f"Open trade reward: {open_trade_reward}\n"
        
        f" REWARD FINAL DEL EPISODIO : {reward}")

        self.prev_action = action
        self.prev_balance = self.balance

        # LÓGICA CORREGIDA PARA EL DICCIONARIO 'info'
        trade_type_to_log = None
        result_to_log = None
        if self.trades:
            last_trade = self.trades[-1]
            trade_type_to_log = last_trade['type']
            result_to_log = last_trade.get('result')
    
        # Construir el diccionario info
        info = {
            'trade_type': step_trade_type,
            'last_trade_type': trade_type_to_log,
            'action': int(action),
            'balance': self.balance,
            'drawdown': self.max_drawdown,
            'profit': last_close_profit if closed_this_step else 0.0,
            'step': self.current_step,
            'result': result_to_log,
            'HOLDS IN POSITION': getattr(self, 'hold_inpos_steps', 0),
            'HOLDS OUT OF POSITION': getattr(self, 'hold_outpos_steps', 0),
            'NOOPS': getattr(self, 'noop_steps', 0)
        }
        if end_reason:
            info['end_reason'] = end_reason
        if terminated or truncated:
            info['episode_trades'] = list(self.trades)
        
        return self._get_observation(), reward, terminated, truncated, info
