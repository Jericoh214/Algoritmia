"""
DATA SET PRINCIPAL 

Proyecto de algoritmia y estructura de datos, acompañado de un resumen ejecutivo

1. 
"""

import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN INICIAL Y SEMILLA
# La semilla garantiza que tu Panel de Gerencia (el docente) vea exactamente 
# los mismos datos que tú viste al hacer las pruebas.
np.random.seed(42) 

# Definimos el volumen de la muestra (N > 1000 según indicaciones)
N = 1500 

# 2. DEFINICIÓN DE NODOS PARA LA RED DE DISTRIBUCIÓN
# Para el Reto 3 (Ford-Fulkerson), necesitarás que Origen y Destino sean nodos de un grafo.
# Creamos una lista representativa de los centros de SafeRoute Logistics.
nodos_logisticos = ['Centro_Lima', 'Centro_Arequipa', 'Centro_Trujillo', 
                    'Puerto_Callao', 'Almacen_Piura', 'Hub_Ica']

# 3. GENERACIÓN ESTOCÁSTICA DE VECTORES
# ID_Producto: Llave primaria, enteros correlativos únicos del 1 al N.
id_producto = np.arange(1, N + 1)

# Volumen_m3: Flotantes uniformes entre 0.5 y 5.0 (Peso para Knapsack)
volumen_m3 = np.random.uniform(0.5, 5.0, N)

# Utilidad_USD: Flotantes numéricos entre 10.0 y 500.0 (Valor para Knapsack)
utilidad_usd = np.random.uniform(10.0, 500.0, N)

# Origen y Destino: Selección aleatoria de nuestra lista de nodos
origen = np.random.choice(nodos_logisticos, N)
destino = np.random.choice(nodos_logisticos, N)

# 4. ENSAMBLAJE DEL DATAFRAME
df_logistica = pd.DataFrame({
    'ID_Producto': id_producto,
    'Volumen_m3': np.round(volumen_m3, 2),   # Redondeamos a 2 decimales por limpieza
    'Utilidad_USD': np.round(utilidad_usd, 2),
    'Origen': origen,
    'Destino': destino
})

# 5. LIMPIEZA LÓGICA DE NEGOCIO
# Un camión no puede tener el mismo nodo de origen y destino.
# Filtramos y corregimos los casos donde Origen == Destino.
for i in range(N):
    while df_logistica.loc[i, 'Origen'] == df_logistica.loc[i, 'Destino']:
        df_logistica.loc[i, 'Destino'] = np.random.choice(nodos_logisticos)

# 6. EVIDENCIA DE EJECUCIÓN
print("--- RESUMEN DEL DATASET GENERADO ---")
print(df_logistica.info())
print("\n--- PRIMEROS 5 REGISTROS ---")
print(df_logistica.head())