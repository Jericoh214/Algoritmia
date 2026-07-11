"""
MÓDULO 1: GENERACIÓN DE DATOS (DATASET ESTOCÁSTICO)
=====================================================
Contiene la función encargada de simular el DataFrame df_logistica
que alimenta a los 4 retos del proyecto (AVL, Knapsack, Flujo Máximo y A*).
"""

import pandas as pd
import numpy as np


def generar_dataset_logistico():
    """
    Esta funcion se encarga de generar un DataFrame simulado para SafeRoute Logistics.

    Se usa una semilla (seed) para que los datos aleatorios sean siempre los
    mismos y la auditoría pueda reproducir el experimento sin errores.

    Complejidad temporal: O(n) — se generan n_registros de forma vectorizada.
    Complejidad espacial: O(n) — se almacenan n_registros en memoria.

    Returns:
        pd.DataFrame: columnas ID_Producto, Volumen_m3, Utilidad_USD,
            Origen, Destino.
    """
    np.random.seed(42) #semilla para datos random (aleatorios o estocásticos)
    n_registros = 2000  # Requisito de la rúbrica: N > 1000

    # Construcción de las columnas usando distribuciones uniformes y aleatorias
    data = {
        "ID_Producto": np.arange(1001, 1001 + n_registros),  # Llaves primarias únicas
        "Volumen_m3": np.round(np.random.uniform(0.5, 15.0, n_registros), 2),  # Peso en Reto 2
        "Utilidad_USD": np.round(np.random.uniform(100.0, 5000.0, n_registros), 2),  # Valor en Reto 2
        "Origen": np.random.randint(1, 5, n_registros),   # Nodos del 1 al 4 (Acopio) -> Reto 3
        "Destino": np.random.randint(6, 10, n_registros),  # Nodos del 6 al 9 (Llegada) -> Reto 3
    }
    return pd.DataFrame(data)