"""
MÓDULO 4: RETO 3 - FORD-FULKERSON (EDMONDS-KARP PARA FLUJO EN REDES)
========================================================================
IMPORTANTE: a diferencia de la versión original, el grafo de capacidades
ya NO está hardcodeado. Se construye dinámicamente a partir de las columnas
Origen y Destino del dataset df_logistica, tal como exige el enunciado
("Origen/Destino actuarán como nodo para el Reto 3").
"""


def construir_grafo_desde_df(df):
    """
    Construye la matriz de capacidades del Reto 3 a partir del dataset real.

    Estrategia: se arma una red en 3 capas con super-origen y super-destino,
    porque el problema tiene múltiples orígenes (nodos 1-4) y múltiples
    destinos (nodos 6-9), y Ford-Fulkerson clásico requiere una única
    fuente y un único sumidero:

        SUPER_ORIGEN -> Nodo_Origen_i -> Nodo_Destino_j -> SUPER_DESTINO

    La capacidad de cada arista Origen_i -> Destino_j es la suma de
    Volumen_m3 de todos los productos que comparten ese par (Origen, Destino).
    Las aristas hacia/desde los nodos super se fijan como la suma total de
    volumen que entra o sale de cada nodo, representando su capacidad máxima
    de despacho/recepción.

    Complejidad temporal: O(n) para recorrer el dataset y agregar volúmenes
    (n = número de filas de df_logistica).
    Complejidad espacial: O(V^2) para la matriz de adyacencia, con
    V = número de nodos (orígenes + destinos + 2 nodos super).

    Args:
        df (pd.DataFrame): dataset con columnas Origen, Destino, Volumen_m3.

    Returns:
        tuple: (grafo_capacidades, node_index, fuente, sumidero)
            grafo_capacidades: matriz NxN de capacidades (float).
            node_index: dict que mapea 'O_<id>' / 'D_<id>' / 'SUPER_ORIGEN' /
                        'SUPER_DESTINO' al índice de fila/columna en la matriz.
            fuente / sumidero: índices de SUPER_ORIGEN y SUPER_DESTINO.
    """
    origenes = sorted(df['Origen'].unique())
    destinos = sorted(df['Destino'].unique())
  #a
    # Asignamos un índice contiguo a cada nodo (necesario para la matriz)
    node_index = {}
    idx = 0
    node_index['SUPER_ORIGEN'] = idx; idx += 1
    for o in origenes:
        node_index[f'O_{o}'] = idx; idx += 1
    for d in destinos:
        node_index[f'D_{d}'] = idx; idx += 1
    node_index['SUPER_DESTINO'] = idx; idx += 1

    n_nodos = idx
    grafo = [[0.0 for _ in range(n_nodos)] for _ in range(n_nodos)]

    # SUPER_ORIGEN -> cada nodo Origen: capacidad = volumen total despachado desde ese origen
    for o in origenes:
        cap = df.loc[df['Origen'] == o, 'Volumen_m3'].sum()
        grafo[node_index['SUPER_ORIGEN']][node_index[f'O_{o}']] += cap

    # Origen -> Destino: capacidad = volumen agregado de todos los envíos de ese par
    agregados = df.groupby(['Origen', 'Destino'])['Volumen_m3'].sum()
    for (o, d), vol in agregados.items():
        grafo[node_index[f'O_{o}']][node_index[f'D_{d}']] += vol

    # Cada nodo Destino -> SUPER_DESTINO: capacidad = volumen total recibido en ese destino
    for d in destinos:
        cap = df.loc[df['Destino'] == d, 'Volumen_m3'].sum()
        grafo[node_index[f'D_{d}']][node_index['SUPER_DESTINO']] += cap

    fuente = node_index['SUPER_ORIGEN']
    sumidero = node_index['SUPER_DESTINO']
    return grafo, node_index, fuente, sumidero


def bfs_caminos_aumentantes(grafo_residual, fuente, sumidero, padres):
    """
    Usa Búsqueda en Anchura (BFS) para encontrar el camino más corto (en saltos)
    desde el Origen (fuente) al Destino (sumidero) que aún tenga capacidad de tráfico.

    Complejidad temporal: O(V^2) en el peor caso al recorrer la matriz de adyacencia.
    Complejidad espacial: O(V) por la cola y el arreglo de visitados.
    """
    visitados = [False] * len(grafo_residual)
    cola = [fuente]  # La cola es la base del BFS (First In, First Out)
    visitados[fuente] = True

    while cola:
        u = cola.pop(0)  # Sacamos el primer nodo que entró a la cola

        # Revisamos todos sus vecinos
        for v, capacidad_disponible in enumerate(grafo_residual[u]):
            if not visitados[v] and capacidad_disponible > 0:
                cola.append(v)
                visitados[v] = True
                padres[v] = u  # Recordamos de dónde venimos para poder trazar la ruta luego

                if v == sumidero:  # ¡Llegamos al puerto! Hay un camino válido
                    return True
    return False  # Ya no hay caminos posibles hacia la meta


def calcular_flujo_maximo(grafo_capacidades, fuente, sumidero):
    """
    Inyecta tráfico por las rutas detectadas por BFS hasta que la red colapse (se llene).
    Implementación de Edmonds-Karp (Ford-Fulkerson con BFS).

    Complejidad temporal: O(V * E^2), cota clásica de Edmonds-Karp
    (V = nodos, E = aristas de la red).
    Complejidad espacial: O(V^2) por la matriz residual.

    Returns:
        tuple: (flujo_maximo_total, cuellos_de_botella) donde cuellos_de_botella
               es una lista de índices (u, v) de aristas saturadas al 100%.
    """
    nodos = len(grafo_capacidades)
    # Matriz para llevar el control de la capacidad que va sobrando
    grafo_residual = [fila[:] for fila in grafo_capacidades]
    padres = [-1] * nodos
    flujo_maximo_total = 0

    # Mientras haya rutas (BFS retorna True)
    while bfs_caminos_aumentantes(grafo_residual, fuente, sumidero, padres):

        flujo_camino = float("Inf")
        s = sumidero
        # 1. Recorremos el camino hacia atrás para encontrar la vía más estrecha (cuello de botella)
        while s != fuente:
            flujo_camino = min(flujo_camino, grafo_residual[padres[s]][s])
            s = padres[s]

        flujo_maximo_total += flujo_camino  # Sumamos este flujo a la exportación total

        v = sumidero
        # 2. Actualizamos la capacidad del camino: restamos capacidad de ida y sumamos la inversa
        while v != fuente:
            u = padres[v]
            grafo_residual[u][v] -= flujo_camino
            grafo_residual[v][u] += flujo_camino
            v = padres[v]

    # 3. Detectamos las vías colapsadas (Empezaron con capacidad > 0 pero ahora tienen 0)
    cuellos_de_botella = []
    for u in range(nodos):
        for v in range(nodos):
            if grafo_capacidades[u][v] > 0 and grafo_residual[u][v] == 0:
                cuellos_de_botella.append((u, v))

    return flujo_maximo_total, cuellos_de_botella


def etiquetas_cuellos_de_botella(cuellos_de_botella, node_index):
    """
    Traduce los pares de índices (u, v) devueltos por calcular_flujo_maximo
    a etiquetas legibles usando node_index (p. ej. 'O_2 -> D_7').
    """
    indice_a_nombre = {v: k for k, v in node_index.items()}
    return [f"{indice_a_nombre[u]} -> {indice_a_nombre[v]}" for u, v in cuellos_de_botella]