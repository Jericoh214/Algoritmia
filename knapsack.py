"""
MÓDULO 3: RETO 2 - KNAPSACK 0/1 (ASIGNACIÓN ÓPTIMA CON PROG. DINÁMICA)
=========================================================================
Determina la combinación exacta de productos que maximiza la utilidad neta
sin exceder la capacidad de carga del camión (Volumen_m3).
"""

import numpy as np


def optimizar_carga_knapsack(df, capacidad_max_m3):
    """
    Llena la mochila (camión) maximizando el valor ($) sin pasar del peso máximo.
    Convertimos los volúmenes decimales a enteros multiplicando por 100 para
    poder indexar la matriz de memoización.

    Complejidad temporal: O(n * W), donde n = número de productos y
    W = capacidad_entera (capacidad discretizada). Es la complejidad
    pseudo-polinomial típica del Knapsack 0/1 por Programación Dinámica.
    Complejidad espacial: O(n * W) por la matriz K completa.

    Args:
        df (pd.DataFrame): dataset con columnas Volumen_m3, Utilidad_USD, ID_Producto.
        capacidad_max_m3 (float): capacidad máxima del camión en m3.

    Returns:
        tuple: (utilidad_maxima, productos_seleccionados, volumen_ocupado)
    """
    # Preparación de datos (Pesos = Volumen, Valores = Dinero)
    pesos = (df['Volumen_m3'] * 100).astype(int).tolist()
    valores = df['Utilidad_USD'].tolist()
    ids = df['ID_Producto'].tolist()

    n = len(valores)
    capacidad_entera = int(capacidad_max_m3 * 100)  # Ej. 50.0m3 -> 5000 unidades

    # Creación de Matriz (Memoización). Filas=Productos, Columnas=Capacidades posibles
    K = [[0.0 for _ in range(capacidad_entera + 1)] for _ in range(n + 1)]

    # Llenado de la matriz (Bottom-Up)
    for i in range(1, n + 1):
        for w in range(1, capacidad_entera + 1):
            if pesos[i - 1] <= w:  # Si el producto cabe en el camión...
                # ¿Qué da más dinero? ¿Meterlo o no meterlo?
                K[i][w] = max(valores[i - 1] + K[i - 1][w - pesos[i - 1]], K[i - 1][w])
            else:  # Si no cabe, copiamos el valor óptimo sin este producto
                K[i][w] = K[i - 1][w]

    utilidad_maxima = K[n][capacidad_entera]

    # Proceso de "Backtracking": Recorrer la tabla hacia atrás para saber qué cargamos
    w_actual = capacidad_entera
    productos_seleccionados = []
    volumen_ocupado = 0.0
    utilidad_restante = utilidad_maxima

    for i in range(n, 0, -1):
        if utilidad_restante <= 0:
            break
        # Si la celda cambió de valor respecto a la de arriba, es que LO METIMOS al camión
        if K[i][w_actual] != K[i - 1][w_actual]:
            productos_seleccionados.append(ids[i - 1])
            utilidad_restante -= valores[i - 1]
            w_actual -= pesos[i - 1]
            volumen_ocupado += (pesos[i - 1] / 100.0)  # Restauramos al valor real con decimales

    return np.round(utilidad_maxima, 2), productos_seleccionados, np.round(volumen_ocupado, 2)