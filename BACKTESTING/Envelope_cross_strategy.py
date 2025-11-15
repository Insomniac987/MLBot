import pandas as pd
import numpy as np
import json
import os
import mplfinance as mpf  # Para graficar velas japonesas
from datetime import datetime

pd.set_option('display.max_rows', None)    # Muestra todas las filas
pd.set_option('display.max_columns', None) # Muestra todas las columnas
pd.set_option('display.width', 0)          # No corta por ancho de pantalla
pd.set_option('display.max_colwidth', None) # Muestra todo el contenido de cada celda
pd.set_option('display.float_format', '{:.2f}'.format)


def backtesting(cruces_upper, cruces_lower,config):
    print("Iniciando backtesting...")

    #Crear carpeta para guardar los gráficos
    output_folder = "img_resultados"
    os.makedirs(output_folder, exist_ok=True)

    #Iniciar la cuenta con un saldo en USD
    saldo_inicial = float(config.get("strategy", {}).get("saldo_inicial"))
    saldo = saldo_inicial
    comision_total = config.get("strategy", {}).get("comision_total")
    timeframe = config.get("common", {}).get("timeframe")
    # evitar_lateral = config.get("strategy", {}).get("evitar_lateral")

    
   
    #Seleccionar el apalancamiento
    apalancamiento = config.get("common", {}).get("apalancamiento")

    # #Obtener el periodo del vol_sma desde config
    # vol_period = config.get("common",{}).get("vol_sma_period")

    # #Obtener el filtro de cuerpo de vela
    # umbral_cuerpo = config.get("strategy", {}).get("umbral_cuerpo_vela")

    #Seleccionar el tipo de moving average a utilizar
    tipo_ma = config.get("strategy", {}).get("tipo_ma")

    match apalancamiento:
      case "1x":
          apalancamiento = 1
      case "5x":
          apalancamiento = 5
      case "10x":
          apalancamiento = 10
      case "25x":
          apalancamiento = 25
      case "50x":
          apalancamiento = 50
      case "100x":
          apalancamiento = 100
      case _:
          print("Opción no válida")

    #Establecer el porcentaje de TP y SL

    percentage_SL = config.get("strategy", {}).get("percentage_SL")
    percentage_TP = config.get("strategy", {}).get("percentage_TP")
       
    #Combinar ambos cruces en un solo conjunto de etiquetas
    cruces_lista = [(idx, 'upper') for idx in cruces_upper] + [(idx, 'lower') for idx in cruces_lower]
    cruces_lista.sort(key=lambda x: x[0]) #Ordenar por indice para procesarlos en orden cronológico

    #Lista para almacenar los resultados de cada operación
    resultados = []

    ##BARRER TODOS LOS CRUCES DE LOS ENVELOPES##
    # operacion_abierta = False
    for idx, tipo_cruce in cruces_lista:
        # if operacion_abierta:
        #    continue
        
        ### VARIABLES ###
        indice_objetivo=None
        indice_salida = None
        razon_salida = None
        indice_cruce = datos.index[idx]
        tipo_mercado = None
        vol_period = config.get("common", {}).get("vol_sma_period")
        filtro_vol = config.get("strategy", {}).get("filtro_vol")
        filtro_cuerpo_vela = config.get("strategy", {}).get("filtro_cuerpo_vela")
        umbral_cuerpo = config.get("strategy", {}).get("umbral_cuerpo_vela")
        
        
        ## FILTRO DE VOLUMEN ##
        # verificar si filtro está activo
        if filtro_vol == True:
          if not filtrar_volumen(vol_period, idx, datos):
            continue

        # ## FILTRO CUERPO DE VELA ##
        if filtro_cuerpo_vela == True:
          if not filtrar_cuerpo_vela(idx,datos,umbral_cuerpo):
            continue
        
        if tipo_cruce == 'upper':
          nivel_cruce = datos['Envelope_Upper_21_6.18'].iloc[idx]
        elif tipo_cruce == 'lower':
          nivel_cruce = datos['Envelope_Lower_21_6.18'].iloc[idx]

        max_velas_futuro = config.get("strategy", {}).get("max_velas_futuro")

        # Buscar en las velas posteriores si alguna cumple que el High >= nivel_cruce
        datos_futuros = datos.iloc[idx+1:idx+max_velas_futuro]  # mirar hasta 500 velas adelante

        nivel_cruce = nivel_cruce + config.get("strategy", {}).get("umbral_cercania")

        for i in datos_futuros.index:
            valor_high = datos_futuros.loc[i, 'High']
            valor_low = datos_futuros.loc[i, 'Low']           

            #Establecer el indice objetivo para cualquier cruce (LONG o SHORT)
            if tipo_cruce == 'lower' and valor_low <= nivel_cruce:
                indice_objetivo = i
                precio_entrada = nivel_cruce
                break
            elif tipo_cruce == 'upper' and valor_high >= nivel_cruce:
                indice_objetivo = i
                precio_entrada = nivel_cruce
                break

        if indice_objetivo is None:
            print(f"❌ No se encontró punto de entrada tras el cruce en {indice_cruce}")
            continue

        #Calcular pendiente en el punto objetivo
        idx_obj = datos.index.get_loc(indice_objetivo)

        #FILTRO: No operar si hay NaN en la media seleccionada
        if tipo_ma == "SMA" and pd.isna(datos[f'SMA_{periodo_sma}'].iloc[idx_obj]):
           print(f"⏩ SMA {periodo_sma} vacío en {indice_objetivo}, se omite entrada")
           continue
        if tipo_ma =="HMA" and pd.isna(datos[f'HMA_{periodo_hma}'].iloc[idx_obj]):
           print(f"⏩ HMA {periodo_hma} vacío en {indice_objetivo}, se omite entrada")
           continue
        if tipo_ma =="EMA" and pd.isna(datos[f'EMA_{periodo_ema}'].iloc[idx_obj]):
           print(f"⏩ EMA {periodo_ema} vacío en {indice_objetivo}, se omite entrada")
           continue



        if tipo_ma == "SMA":
           pendiente = calcular_pendiente_sma(datos, idx_obj, n=10)
        elif tipo_ma == 'HMA':
           pendiente = calcular_pendiente_hma(datos, idx_obj, n=10)
        elif tipo_ma == 'EMA':
           pendiente = calcular_pendiente_ema(datos, idx_obj, n=10)
        
        #Obtener el umbral de la pendiente para determinar el tipo de mercado
        umbral_pendiente = config.get("strategy", {}).get("umbral_pendiente")

        #Obtener valores del RSI
        rsi_min = config.get("strategy", {}).get("rsi_min")
        rsi_max = config.get("strategy", {}).get("rsi_max")
        rsi_index_label =datos.index[idx]
        rsi_valor= datos.loc[rsi_index_label, "RSI"]

        #Determinar el tipo de mercado y definir TP/SL en base al cruce y mercado
                    
        ####SI EL MERCADO ES ALCISTA (SMA APUNTANDO HACIA ARRIBA)
        
        if pendiente > umbral_pendiente:
          
          ### REVERTIR ###
          if filtro_rsi_inverso == True:
            if rsi_valor < rsi_max:
              tipo_mercado = "MERCADO BAJISTA"
              direccion_operacion = "SHORT"
              print(f"⚠️ RSI={rsi_valor} indica sobrecompra. Revirtiendo LONG ➡️ SHORT")
              take_profit = precio_entrada * ( 1 - percentage_TP)
              stop_loss = precio_entrada * ( 1 + percentage_SL)
            else:
              tipo_mercado = "MERCADO ALCISTA"
              direccion_operacion = "LONG"
              take_profit = precio_entrada * ( 1 + percentage_TP)
              stop_loss = precio_entrada * ( 1 - percentage_SL)
          else:
              tipo_mercado = "MERCADO ALCISTA"
              direccion_operacion = "LONG"
              take_profit = precio_entrada * ( 1 + percentage_TP)
              stop_loss = precio_entrada * ( 1 - percentage_SL)

        elif pendiente < -umbral_pendiente:

          if filtro_rsi_inverso == True:
            if rsi_valor > rsi_min:
              tipo_mercado = "MERCADO ALCISTA"
              direccion_operacion = "LONG"
              print(f"⚠️ RSI={rsi_valor} indica sobreventa. Revirtiendo SHORT ➡️ LONG")
              take_profit = precio_entrada * ( 1 + percentage_TP)
              stop_loss = precio_entrada * ( 1 - percentage_SL) 
            else:
              tipo_mercado = "MERCADO BAJISTA"
              direccion_operacion = "SHORT"
              take_profit = precio_entrada * ( 1 - percentage_TP)
              stop_loss = precio_entrada * ( 1 + percentage_SL)
          else:
              tipo_mercado = "MERCADO BAJISTA"
              direccion_operacion = "SHORT"
              take_profit = precio_entrada * ( 1 - percentage_TP)
              stop_loss = precio_entrada * ( 1 + percentage_SL)

        else:

          tipo_mercado = "MERCADO LATERAL"
          if tipo_cruce == 'lower':
            if filtro_rsi_inverso == True:
              if rsi_valor > rsi_max:
                direccion_operacion = "SHORT"
                print(f"⚠️ RSI={rsi_valor} indica sobrecompra. Revirtiendo LONG ➡️ SHORT")
                take_profit = precio_entrada * ( 1 -percentage_TP)
                stop_loss = precio_entrada * ( 1 + percentage_SL)
              else:
                direccion_operacion = "LONG"
                take_profit = precio_entrada * ( 1 + percentage_TP)
                stop_loss = precio_entrada * ( 1 - percentage_SL)
            else:
                direccion_operacion = "LONG"
                take_profit = precio_entrada * ( 1 + percentage_TP)
                stop_loss = precio_entrada * ( 1 - percentage_SL)
          else:
            if filtro_rsi_inverso == True:
              if rsi_valor < rsi_min:
                print(f"⚠️ RSI={rsi_valor} indica sobreventa. Revirtiendo SHORT ➡️ LONG")
                direccion_operacion = "LONG"
                take_profit = precio_entrada * ( 1 + percentage_TP)
                stop_loss = precio_entrada * ( 1 - percentage_SL) 
              else:  
                direccion_operacion = "SHORT"
                take_profit = precio_entrada * ( 1 -percentage_TP)
                stop_loss = precio_entrada * ( 1 + percentage_SL)
            else:  
                direccion_operacion = "SHORT"
                take_profit = precio_entrada * ( 1 -percentage_TP)
                stop_loss = precio_entrada * ( 1 + percentage_SL)

        ##Una vez que se tiene el precio de entrada, SL y TP, ver en qué momento se cumple el TP y/o SL
        print (f"Fecha cruce: {indice_cruce}, Fecha objetivo: {indice_objetivo}")
        # operacion_abierta = True

        for j in datos_futuros.loc[datos_futuros.index > indice_objetivo].index:        
                
            valor_high = datos_futuros.loc[j, 'High']
            valor_low = datos_futuros.loc[j, 'Low']  

            # Detectar salida según tipo de mercado y tipo de cruce (con TP/SL)
            # Mercados LATERALES
            if direccion_operacion == "LONG":
                if valor_high >= take_profit:
                    indice_salida = j
                    precio_salida = take_profit
                    razon_salida = "Take profit"
                    break
                elif valor_low <= stop_loss:
                    indice_salida = j
                    precio_salida = stop_loss
                    razon_salida = "SL Exit"
                    break
            elif direccion_operacion == "SHORT":
                if valor_low <= take_profit:
                    indice_salida = j
                    precio_salida = take_profit
                    razon_salida = "Take profit"
                    break
                elif valor_high >= stop_loss:
                    indice_salida = j
                    precio_salida = stop_loss
                    razon_salida = "SL Exit"
                    break

        if indice_salida is not None:

            # 1. Calcular el cambio porcentual de precio
            if direccion_operacion == 'LONG':
                resultado_pct = (precio_salida - precio_entrada) / precio_entrada
            else:  # SHORT
                resultado_pct = (precio_entrada - precio_salida) / precio_entrada
            
            # 2. Restar comisión total (ida + vuelta)
            resultado_pct -= comision_total  # <- esto ya está en porcentaje
            
            # 3. Calcular el resultado total con apalancamiento
            resultado = saldo * (resultado_pct * apalancamiento)
            
            # 4. Actualizar el saldo
            saldo += resultado  # no usar *=, eso mezcla porcentajes con montos absolutos
              #Imprimir en verde si fué ganadora, en rojo si fué perdedora
            if razon_salida == 'Take profit':
                print(f"\033[92m***{tipo_mercado}***-->Operación ganadora ({razon_salida} en {indice_salida}: Resultado={resultado:.2f}: Saldo Actual:{saldo}:\033[0m")
            else:
                print(f"\033[91m***{tipo_mercado}***Operación perdedora ({razon_salida} en {indice_salida}: Resultado={resultado:.2f}: Saldo Actual:{saldo}\033[0m)")
            # operacion_abierta = False
            #guardar el resultado
            resultados.append({
                'Tipo': 'Long' if direccion_operacion == 'LONG' else 'Short',
                'Fecha Objetivo': indice_objetivo,
                'Fecha Cruce': indice_cruce,
                'Precio_Entrada': precio_entrada,
                'Fecha_Salida': indice_salida,
                'Precio Salida': precio_salida,
                'Resultado': resultado,
                'Razon Salida': razon_salida,
                'Balance': saldo,
                'Tipo de Mercado': tipo_mercado
            })

            ####GRAFICOS####
            graficar = config.get("strategy", {}).get("graficar")

            if graficar is not False:

              #Extender el rango de fechas para graficar y definir variables
              fecha_inicio = indice_cruce - pd.Timedelta(hours=120)  # Extiende 48 horas hacia atrás
              fecha_salida = indice_salida + pd.Timedelta(hours=120) # Extiende 'N' horas hacia adelante
              rango_total = datos.loc[fecha_inicio:fecha_salida]
  
  
              # Crear la línea horizontal desde el cruce hasta que el precio lo toque
              linea_objetivo = pd.Series(index=rango_total.index, data=np.nan)
              if indice_objetivo in linea_objetivo.index:
                linea_objetivo.loc[indice_cruce:indice_objetivo] = precio_entrada
  
              #Crear una serie para el scatter en el punto de objetivo
              scatter_objetivo = pd.Series(index=rango_total.index, data=np.nan)
              if indice_objetivo in scatter_objetivo.index:
                scatter_objetivo.loc[indice_objetivo] = precio_entrada #Marcar el nivel de entrada
  
              #Crear una serie para el scatter en el punto de objetivo
              scatter_cruce = pd.Series(index=rango_total.index, data=np.nan)
              if indice_cruce in scatter_cruce.index:
                scatter_cruce.loc[indice_cruce] = nivel_cruce #Marcar el nivel de entrada
  
              # Crear la línea horizontal desde el cruce hasta que el precio lo toque
              linea_TP = pd.Series(index=rango_total.index, data=np.nan)
              if indice_objetivo in linea_TP.index:
                linea_TP.loc[indice_objetivo:indice_salida] = take_profit
  
              # Crear la línea horizontal desde el cruce hasta que el precio lo toque
              linea_SL = pd.Series(index=rango_total.index, data=np.nan)
              if indice_objetivo in linea_SL.index:
                linea_SL.loc[indice_objetivo:indice_salida] = stop_loss
  
              # Crear la serie para la sma
              if f'SMA_{periodo_sma}' in rango_total.columns:
                 sma = rango_total[f'SMA_{periodo_sma}']
                 #Solo agregar si hay al menos un dato válido
                 if sma.dropna().empty:
                    sma = None
              else:
                 sma = None
  
              # Crear la serie para la hma
              if f'HMA_{periodo_hma}' in rango_total.columns:
                 hma = rango_total[f'HMA_{periodo_hma}']
                 #Solo agregar si hay al menos un dato válido
                 if hma.dropna().empty:
                    hma = None
              else:
                 hma = None
  
              scatter_TP = pd.Series(index=rango_total.index, data=np.nan)
              if indice_salida in scatter_TP.index:
                 scatter_TP.loc[indice_salida]= take_profit
              scatter_SL = pd.Series(index=rango_total.index, data=np.nan)
              if indice_salida in scatter_SL.index:
                scatter_SL.loc[indice_salida] = stop_loss
              # Crear los elementos para graficar
              add_plots = [
                  mpf.make_addplot(rango_total['Envelope_Upper_21_6.18'], color='magenta', label='Envelope Upper (21, 6.18%)'),
                  mpf.make_addplot(rango_total['Envelope_Lower_21_6.18'], color='magenta', label='Envelope Lower (21, 6.18%)'),
                  mpf.make_addplot(rango_total['Envelope_Upper_55_5'], color='green', label='Envelope Upper (55, 5%)'),
                  mpf.make_addplot(rango_total['Envelope_Lower_55_5'], color='green', label='Envelope Lower (55, 5%)'),
                  mpf.make_addplot(linea_objetivo, linestyle ='--', color='orange', label="linea objetivo"),
                  mpf.make_addplot(scatter_cruce, type='scatter', markersize=50, color = 'purple', label="Scatter Cruce"),
                  mpf.make_addplot(scatter_objetivo, type='scatter', markersize=50, color='blue',label = "Scatter Objetivo"),
                  mpf.make_addplot(linea_TP if razon_salida== 'Take profit' else linea_SL, linestyle='--', color='green' if razon_salida== 'Take profit' else "red", label='Línea TP / SL'),
                  mpf.make_addplot(scatter_TP if razon_salida == 'Take profit' else scatter_SL, type='scatter', color='green' if razon_salida == 'Take profit' else 'red', markersize=50, label="Scatter TP /SL")        
                  ]
              
              if sma is not None and tipo_ma == "SMA":
                 add_plots.append(mpf.make_addplot(sma, color ='blue', label=f'SMA {config.get("common", {}).get("sma_period")}'))
  
              elif hma is not None and tipo_ma == "HMA":
                 add_plots.append(mpf.make_addplot(hma, color ='blue', label= f'HMA {config.get("common",{}).get("hma_period")}'))
              
              #Título del gráfico
              titulo = f"{'Long' if tipo_cruce == 'lower' else 'Short'} | Resultado: {resultado:.2f} | {razon_salida} | Balance Actual: {saldo} | Tipo de mercado: {tipo_mercado}"
              #Guardar el gráfico como PNG
              output_file = os.path.join(output_folder, f"resultado_{indice_cruce.strftime('%Y%m%d_%H%M%S')}.png")
              # Graficar el resultado
              mpf.plot(
                  rango_total,
                  type='candle',
                  style='charles',
                  title=titulo,
                  ylabel='Precio',
                  addplot=add_plots,
                  volume=False,
                  figsize=(16, 9),
                  savefig= output_file
              )
              print(f"Gráfico guardado: {output_file}")
            
            else:
             print(f"La opción de graficar está en modo {graficar}. No se creará el gráfico")

        else:
            print(f"No se alcanzó ni TP ni SL para el cruce en {indice_cruce}.")
    
    resultados_df = pd.DataFrame(resultados)

    #Calcular el saldo final
    saldo_final = saldo

    #Calcular el balance máximo y mínimo durante el backtesting
    balance_max = resultados_df['Balance'].max()
    balance_min = resultados_df['Balance'].min()
    print(f"\nBalance máximo durante el backtesting: {balance_max}")
    print(f"Balance mínimo durante el backtesting: {balance_min}")

    #Verificar que resultados no esté vacío
    if resultados_df.empty:
       print("No se registraron operaciones en el backtesting.")
       return

    #Calcular efectividad
    ganadoras = resultados_df[resultados_df['Razon Salida'] == 'Take profit'].shape[0]
    total = resultados_df.shape[0]
    print(f"Ganadoras: {ganadoras} Total: {total}")
    efectividad = (ganadoras/total) * 100 if total>0 else 0

    print("\nResultados del Backtesting:")
    print(resultados_df)
    print(f"\nSaldo inicial: {saldo_inicial}")
    print(f"\nSaldo Final: {saldo_final}")
    print(f"\nEfectividad de la estrategia: {efectividad:.2f}%")

    ### EFECTIVIDAD POR TIPO DE MERCADO ###
    print("\nEfectividad por tipo de mercado")
    efectividad_mercado = resultados_df.groupby("Tipo de Mercado").apply(
       lambda df: 100 * (df['Razon Salida'] == 'Take profit').sum() / len(df)
    )
    print(efectividad_mercado.round(2).astype(str) + " %")

    ### RESUMEN DE RESULTADOS POR TIPO DE MERCADO ####
    print("\nResumen de resultados por tipo de mercado:")
    resumen = resultados_df.groupby(['Tipo de Mercado', 'Razon Salida']).size().unstack(fill_value=0)
    resumen['Total'] = resumen.sum(axis=1)

    #Si faltan columnas, las agrega con 0
    for col in ['Take profit', 'SL Exit']:
       if col not in resumen.columns:
          resumen[col]=0
    resumen = resumen[['Take profit', 'SL Exit', 'Total']]
    print(resumen)

    ###Mostrar efectividad y resumen por tipo de mercado###
    print("\nEfectividad y resumen por tipo de mercado:")
    tabla = resumen.copy()
    tabla['Efectividad (%)'] = efectividad_mercado.round(2)
    print(tabla) 

    ###GUARDAR RESULTADOS###

    output_resultados= "registro_resultados"
    os.makedirs(output_resultados, exist_ok=True)

    # resultados_df.to_csv(f'resultados_backtesting_{efectividad:.2f}_{apalancamiento}_{timeframe}.csv', index=False)

    timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    filename=f"resultados_backtesting_{timestamp}_{efectividad:.2f}_{apalancamiento}x_{timeframe}.txt"

    with open(os.path.join(output_resultados, filename), 'w') as f:
      f.write("RESULTADOS:\n")
      f.write(f"EFECTIVIDAD DE LA ESTRATEGIA: {efectividad:.2f}%\n")
      f.write(f"SALDO INICIAL: {saldo_inicial} usd\n")
      f.write(f"SALDO FINAL: {saldo_final}usd\n")
      f.write(f"SALDO MÁXIMO LOGRADO: {balance_max}\n")
      f.write(f"SALDO MÍNIMO: {balance_min}\n")

      f.write("CONFIGURACIÓN USADA:\n")
      json.dump(config, f, indent=2)
      f.write("\n\nRESULTADOS A DETALLE:\n")
      f.write(resultados_df.to_string(index=False))

    return resultados_df

def calcular_pendiente_sma(df, idx, n=10):
  if idx < n:
    return np.nan
  precio_actual = df[f'SMA_{periodo_sma}'].iloc[idx]
  precio_anterior = df[f'SMA_{periodo_sma}'].iloc[idx-n]
  pendiente = (precio_actual - precio_anterior) / precio_anterior
  return pendiente

def calcular_pendiente_hma(df, idx, n=10):
  if idx < n:
    return np.nan
  precio_actual = df[f'HMA_{periodo_hma}'].iloc[idx]
  precio_anterior = df[f'HMA_{periodo_hma}'].iloc[idx-n]
  pendiente = (precio_actual - precio_anterior) / precio_anterior
  return pendiente

def calcular_pendiente_ema(df, idx, n=10):
  if idx < n:
    return np.nan
  precio_actual = df[f'EMA_{periodo_ema}'].iloc[idx]
  precio_anterior = df[f'EMA_{periodo_ema}'].iloc[idx-n]
  pendiente = (precio_actual - precio_anterior) / precio_anterior
  return pendiente

def filtrar_cercanos(cercanos, cruces, N):
   #Solo deja los cercanos que NO estén a menos de N velas de un cruce
   return np.array([
      idx for idx in cercanos
      if not np.any(np.abs(cruces - idx) <= N)
   ])

def filtrar_pegados(indices, N):
   indices = np.sort(indices)
   resultado = []
   ultimo = -N*2
   for idx in indices:
      if idx - ultimo > N:
         resultado.append(idx)
         ultimo = idx
   return np.array(resultado)

def filtrar_volumen(vol_period, idx, df):
    col_vol = f"VOL_SMA_{vol_period}" 
    if col_vol in df.columns:
        try:
            # Convertir idx (entero) a etiqueta real del índice
            index_label = df.index[idx]
            volumen_actual = df.loc[index_label, 'Volumen']
            volumen_ma = df.loc[index_label, col_vol]
            if volumen_actual < volumen_ma:
                print(f"⏩ Entrada descartada por volumen bajo en {index_label}: Vol {volumen_actual:.2f} < MA({vol_period}) {volumen_ma:.2f}")
                return False
            return True
        except Exception as e:
            print(f"Error al evaluar el volumen {idx}: {e}")
            return False
    else:
        print(f"⚠️ Columna de volumen {col_vol} no encontrada. Skipping")
        return False

def filtrar_cuerpo_vela(idx, df, umbral_cuerpo):
  try:
     index_label = df.index[idx]
     open_ = df.loc[index_label]['Open']
     close = df.loc[index_label]['Close']
     high = df.loc[index_label]['High']
     low = df.loc[index_label]['Low']

     cuerpo = abs(close - open_)
     rango = high -low
     #Evita división entre cero
     cuerpo_pct = cuerpo / rango if rango > 0 else 0

     if cuerpo_pct < umbral_cuerpo:
        print(f"⏩ Entrada descartada por vela indecisa ({cuerpo_pct:.2%} cuerpo) en {index_label}")
        return False
     return True
  except Exception as e:
     print(f"Error en filtro de cuerpo de vela: {e}")
     return False
  
def calcular_rsi(df, periodo=14):
   delta = df['Close'].diff()
   ganancia = delta.clip(lower=0)
   perdida = -delta.clip(upper=0)

   avg_ganancia = ganancia.rolling(window=periodo).mean()
   avg_perdida = perdida.rolling(window=periodo).mean()

   rs = avg_ganancia / avg_perdida
   rsi = 100 - (100 / (1 + rs))
   return rsi
    
####   M   A   I   N   ###

if __name__ == "__main__":

    # Ruta al archivo Excel
    cargar_datos = 'Data.csv'

    #Configurar el json de las variables
    with open("config_dev.json", "r") as f:
      config = json.load(f)
    # Leer archivo
    try:
        datos = pd.read_csv(cargar_datos)
        print("Archivo de Excel cargado correctamente.")
    except FileNotFoundError:
        print(f"No se encontró el archivo: {cargar_datos}")
        exit()
    except Exception as e:
        print("Error al leer el archivo:", e)
        exit()

    # Renombrar columnas para mplfinance
    datos.rename(columns={
        'Timestamp': 'Date',
        'Precio de Apertura': 'Open',
        'Precio Máximo': 'High',
        'Precio Mínimo': 'Low',
        'Precio de Cierre': 'Close'
    }, inplace=True)

    # Convertir fecha y establecer índice
    datos['Fecha/Hora de Apertura'] = pd.to_datetime(datos['Fecha/Hora de Apertura'])
    datos.set_index('Fecha/Hora de Apertura', inplace=True)

    # Calcular RSI
    datos["RSI"]= calcular_rsi(datos, periodo=14)

    #Establecer un umbral de cercanía para señales
    umbral_cercania = float(config.get("strategy", {}).get("umbral_cercania")) / 100 #convierte 1 a 0.01, 10 a 0.10

      # Calcular diferencias de envelopes
    diff_upper = datos['Envelope_Upper_21_6.18'] - datos['Envelope_Upper_55_5']
    diff_lower = datos['Envelope_Lower_21_6.18'] - datos['Envelope_Lower_55_5']

    # Detectar cruces
    cruces_upper = np.where((diff_upper.shift(1) < 0) & (diff_upper > 0))[0]
    cruces_lower = np.where((diff_lower.shift(1) > 0) & (diff_lower < 0))[0]

    N = config.get("strategy", {}).get("umbral_pegado") #Número de velas de separación mínima respecto a un cruce

    # ---NUEVO---: Detectar cercanía de medias ---
    # cercanos_upper = np.where(diff_upper.abs() < umbral_cercania)[0]
    # cercanos_lower = np.where(diff_lower.abs() < umbral_cercania)[0]

    cercanos_upper = np.where((diff_upper.abs()/ datos['Envelope_Upper_55_5']) < umbral_cercania)[0]
    cercanos_lower = np.where((diff_lower.abs()/datos['Envelope_Lower_55_5']) < umbral_cercania)[0]

    # ---FILTRO 1: eliminar cercanías muy pegadas a un cruce ---

    cercanos_upper_filtrados = filtrar_cercanos(cercanos_upper,cruces_upper, N)
    cercanos_lower_filtrados = filtrar_cercanos(cercanos_lower, cruces_lower, N)

    # ----FILTRO 2: Eliminar señales cercanas entre sí (solo una por zona) ---

    cercanos_upper_filtrados = filtrar_pegados(cercanos_upper_filtrados, N)
    cercanos_lower_filtrados = filtrar_pegados(cercanos_lower_filtrados, N)


    #Unir señales de cruce y cercanía (sin duplicados)
    indices_upper = np.unique(np.concatenate([cruces_upper, cercanos_upper_filtrados]))
    indices_lower = np.unique(np.concatenate([cruces_lower, cercanos_lower_filtrados]))


    # DataFrame de cruces
    cruces = pd.DataFrame(index=datos.index)
    cruces['Cruces Upper'] = np.nan
    cruces['Cruces Lower'] = np.nan

    indices_upper = indices_upper.astype(int)
    indices_lower = indices_lower.astype(int)

    cruces.iloc[indices_upper, cruces.columns.get_loc('Cruces Upper')] = datos['Envelope_Upper_21_6.18'].iloc[indices_upper]
    cruces.iloc[indices_lower, cruces.columns.get_loc('Cruces Lower')] = datos['Envelope_Lower_21_6.18'].iloc[indices_lower]

    # Ejecutar estrategia
    # backtesting(indices_upper, indices_lower, config)

    # ########BLOQUE DE COMPARACIONES########

    filtro_vol = config.get("strategy", {}).get("filtro_vol")
    filtro_cuerpo_vela = config.get("strategy", {}).get("filtro_cuerpo_vela")
    filtro_rsi_inverso = config.get("strategy", {}).get("filtro_rsi_inverso")

    combinaciones_tp_sl = config.get("strategy", {}).get("combinaciones_tp_sl", [])
    pendientes = config.get("strategy", {}).get("umbral_pendiente_lista")
    combinaciones_sma = config.get("common", {}).get("sma_period_lista")
    combinaciones_hma = config.get("common", {}).get("hma_period_lista")
    combinaciones_ema = config.get("common", {}).get("ema_period_lista")
    if filtro_rsi_inverso == True:
      combinaciones_rsi = config.get("strategy", {}).get("combinaciones_rsi", [])
    else:
      combinaciones_rsi = [(None, None)]
    if filtro_vol == True:
      combinaciones_vol_sma = config.get("common", {}).get("vol_sma_period_lista")
    else:
      combinaciones_vol_sma = [None]
    if filtro_cuerpo_vela == True:
      combinaciones_umbral_cuerpo_vela = config.get("strategy", {}).get("umbral_cuerpo_vela_lista")
    else:
      combinaciones_umbral_cuerpo_vela = [None]
    tipo_ma = config.get("strategy", {}).get("tipo_ma", '')



    resumen_resultados = []

    if tipo_ma == "SMA":
      for sma in combinaciones_sma: 
        for pendiente in pendientes:
          for tp, sl in combinaciones_tp_sl:   
            for vol in combinaciones_vol_sma:
              for cuerpo_vela in combinaciones_umbral_cuerpo_vela:
                for rsi_min, rsi_max in combinaciones_rsi:
                  config["strategy"]["percentage_TP"] = tp
                  config["strategy"]["percentage_SL"] = sl
                  config["strategy"]["umbral_pendiente"] = pendiente
                  config["strategy"]["rsi_min"] = rsi_min
                  config["strategy"]["rsi_max"] = rsi_max
                  config["common"]["sma_period"] = sma
                  if filtro_vol == True:
                    config["common"]["vol_sma_period"] = vol
                  else:
                    config["common"]["vol_sma_period"] = None
                  if filtro_cuerpo_vela == True:
                    config["strategy"]["umbral_cuerpo_vela"] = cuerpo_vela
                  else:
                    config["strategy"]["umbral_cuerpo_vela"] = None 
                  if filtro_rsi_inverso == True:
                    config["strategy"]["rsi_min"] = rsi_min
                    config["strategy"]["rsi_max"] = rsi_max
                  else:
                    config["strategy"]["rsi_min"] = None
                    config["strategy"]["rsi_max"] = None
                  config["strategy"]["saldo_inicial"] = 200
                  print(f"🔍 Probando: TP={tp}, SL={sl}, Umbral={pendiente}")
                  global periodo_sma
                  periodo_sma = sma
                  resultados = backtesting(indices_upper, indices_lower, config)
                  if resultados is not None:
                    ganadoras = resultados[resultados['Razon Salida'] == 'Take profit'].shape[0]
                    total = resultados.shape[0]
                    efectividad = (ganadoras / total) * 100 if total > 0 else 0
                    saldo_final = resultados.iloc[-1]['Balance'] if not resultados.empty else 0
                    resumen_resultados.append({
                    'SMA': sma,
                    'Vol_SMA': vol,
                    'cuerpo_vela': cuerpo_vela,
                    'TP': tp,
                    'SL': sl,
                    'Pendiente': pendiente * 100,
                    'Operaciones': total,
                    'Efectividad (%)': round(efectividad, 2),
                    'Saldo Final': round(saldo_final, 2)
                    })
    elif tipo_ma == "HMA":
      for hma in combinaciones_hma: 
        for pendiente in pendientes:
          for tp, sl in combinaciones_tp_sl:
            for vol in combinaciones_vol_sma:
              for cuerpo_vela in combinaciones_umbral_cuerpo_vela:
                for rsi_min, rsi_max in combinaciones_rsi:
                  config["strategy"]["percentage_TP"] = tp
                  config["strategy"]["percentage_SL"] = sl
                  config["strategy"]["umbral_pendiente"] = pendiente
                  config["strategy"]["rsi_min"] = rsi_min
                  config["strategy"]["rsi_max"] = rsi_max
                  if filtro_cuerpo_vela == True:
                    config["strategy"]["umbral_cuerpo_vela"] = cuerpo_vela
                  else:
                    config["strategy"]["umbral_cuerpo_vela"] = None 
                  if filtro_rsi_inverso == True:
                    config["strategy"]["rsi_min"] = rsi_min
                    config["strategy"]["rsi_max"] = rsi_max
                  else:
                    config["strategy"]["rsi_min"] = None
                    config["strategy"]["rsi_max"] = None
                  config["strategy"]["saldo_inicial"] = 200
                  config["common"]["hma_period"] = hma
                  if filtro_vol == True:
                    config["common"]["vol_sma_period"] = vol
                  else:
                    config["common"]["vol_sma_period"] = None
                  config["strategy"]["saldo_inicial"] = 200
                  print(f"🔍 Probando: TP={tp}, SL={sl}, Umbral={pendiente}")
                  global periodo_hma
                  periodo_hma = hma
                  resultados = backtesting(indices_upper, indices_lower, config)
                  if resultados is not None:
                    ganadoras = resultados[resultados['Razon Salida'] == 'Take profit'].shape[0]
                    total = resultados.shape[0]
                    efectividad = (ganadoras / total) * 100 if total > 0 else 0
                    saldo_final = resultados.iloc[-1]['Balance'] if not resultados.empty else 0
                    resumen_resultados.append({
                    'HMA': hma,
                    'Vol_SMA': vol,
                    # 'cuerpo_vela': cuerpo_vela,
                    'TP': tp,
                    'SL': sl,
                    'Pendiente': pendiente * 100,
                    'Operaciones': total,
                    'Efectividad (%)': round(efectividad, 2),
                    'Saldo Final': round(saldo_final, 2)
                    })
    elif tipo_ma == "EMA":
      for ema in combinaciones_ema: 
        for pendiente in pendientes:
          for tp, sl in combinaciones_tp_sl:
            for vol in combinaciones_vol_sma:
              for cuerpo_vela in combinaciones_umbral_cuerpo_vela:
                for rsi_min, rsi_max in combinaciones_rsi:
                  config["strategy"]["percentage_TP"] = tp
                  config["strategy"]["percentage_SL"] = sl
                  config["strategy"]["umbral_pendiente"] = pendiente
                  config["strategy"]["rsi_min"] = rsi_min
                  config["strategy"]["rsi_max"] = rsi_max
                  if filtro_cuerpo_vela == True:
                    config["strategy"]["umbral_cuerpo_vela"] = cuerpo_vela
                  else:
                    config["strategy"]["umbral_cuerpo_vela"] = None 
                  config["strategy"]["saldo_inicial"] = 200
                  config["common"]["ema_period"] = ema
                  if filtro_vol == True:
                    config["common"]["vol_sma_period"] = vol
                  else:
                    config["common"]["vol_sma_period"] = None
                  if filtro_rsi_inverso == True:
                    config["strategy"]["rsi_min"] = rsi_min
                    config["strategy"]["rsi_max"] = rsi_max
                  else:
                    config["strategy"]["rsi_min"] = None
                    config["strategy"]["rsi_max"] = None
                  config["strategy"]["saldo_inicial"] = 200
                  print(f"🔍 Probando: TP={tp}, SL={sl}, Umbral={pendiente}")
                  global periodo_ema
                  periodo_ema = ema
                  resultados = backtesting(indices_upper, indices_lower, config)
                  if resultados is not None:
                    ganadoras = resultados[resultados['Razon Salida'] == 'Take profit'].shape[0]
                    total = resultados.shape[0]
                    efectividad = (ganadoras / total) * 100 if total > 0 else 0
                    saldo_final = resultados.iloc[-1]['Balance'] if not resultados.empty else 0
                    resumen_resultados.append({
                    'EMA': ema,
                    'Vol_SMA': vol,
                    # 'cuerpo_vela': cuerpo_vela,
                    'TP': tp,
                    'SL': sl,
                    'Pendiente': pendiente * 100,
                    'Operaciones': total,
                    'Efectividad (%)': round(efectividad, 2),
                    'Saldo Final': round(saldo_final, 2)
                    })
       
    # Mostrar resumen solo si hay resultados
    if resumen_resultados:
      df_resumen = pd.DataFrame(resumen_resultados).sort_values(by="Efectividad (%)", ascending=False)
      print("\n\U0001F4CA Tabla comparativa de resultados:")
      print(df_resumen.to_string(index=False))
    else:
      print("\n⚠️ No se registraron resultados para comparar. Revisa si la lista 'pendientes' está vacía o si el backtesting no generó operaciones.")