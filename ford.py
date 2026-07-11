"""
MÓDULO 4: RETO 3 - FORD-FULKERSON (EDMONDS-KARP PARA FLUJO EN REDES)
========================================================================
IMPORTANTE: a diferencia de la versión original, el grafo de capacidades
ya NO está hardcodeado. Se construye dinámicamente a partir de las columnas
Origen y Destino del dataset df_logistica, tal como exige el enunciado
("Origen/Destino actuarán como nodo para el Reto 3").

NOTA SOBRE EL MODELADO: las aristas Origen_i -> Destino_j representan
carreteras con una CAPACIDAD FÍSICA propia (simulada, ver
generar_capacidad_fisica_infraestructura), independiente de cuánta demanda
exista. Si la capacidad de las aristas se derivara únicamente de la
demanda agregada del dataset, la red siempre terminaría saturada al 100%
en todas partes por construcción (no habría cuellos de botella reales,
solo una tautología). Separar demanda de capacidad física es lo que
permite que Edmonds-Karp encuentre cuellos de botella genuinos.
"""

import numpy as np


def generar_capacidad_fisica_infraestructura(origenes, destinos, capacidad_min=300.0,
                                               capacidad_max=700.0, seed=7):
    """
    Simula el límite físico de cada carretera (Origen_i -> Destino_j),
    independiente de la demanda del dataset. Representa una restricción de
    infraestructura real (ancho de la vía, límite de puente, permisos de
    carga) que no depende de cuántos productos se quieran despachar ese día.

    El PDF no provee datos reales de capacidad vial, así que se simula con
    una distribución uniforme reproducible (semilla fija) para que la
    auditoría sea consistente entre ejecuciones. En un caso de negocio real,
    estos valores vendrían de un catastro de infraestructura vial de la
    empresa, no del dataset de pedidos.

    Complejidad temporal: O(O * D), con O = número de orígenes y
    D = número de destinos (se genera una capacidad por cada par posible).
    Complejidad espacial: O(O * D) por el diccionario resultante.

    Returns:
        dict: {(origen, destino): capacidad_fisica_m3} para cada par posible.
    """
    rng = np.random.default_rng(seed)
    capacidad_fisica = {}
    for o in origenes:
        for d in destinos:
            capacidad_fisica[(o, d)] = round(float(rng.uniform(capacidad_min, capacidad_max)), 2)
    return capacidad_fisica


def construir_grafo_desde_df(df, capacidad_fisica=None):
    """
    Construye la matriz de capacidades del Reto 3 a partir del dataset real.

    Estrategia: se arma una red en 3 capas con super-origen y super-destino,
    porque el problema tiene múltiples orígenes (nodos 1-4) y múltiples
    destinos (nodos 6-9), y Ford-Fulkerson clásico requiere una única
    fuente y un único sumidero:

        SUPER_ORIGEN -> Nodo_Origen_i -> Nodo_Destino_j -> SUPER_DESTINO

    Las aristas SUPER_ORIGEN -> Origen_i y Destino_j -> SUPER_DESTINO
    representan DEMANDA: la suma de Volumen_m3 que cada origen despacha o
    cada destino recibe según el dataset.

    Las aristas Origen_i -> Destino_j representan CAPACIDAD FÍSICA de la
    carretera (ver generar_capacidad_fisica_infraestructura), NO la demanda
    de ese par. Esto es intencional: cuando la demanda agregada de un par
    (Origen, Destino) supera la capacidad física simulada de esa carretera,
    esa arista se convierte en un cuello de botella real y mide menos del
    100% de flujo posible respecto a lo que se necesitaría; cuando la
    capacidad física sobra, la arista queda sub-utilizada. Esa variación es
    precisamente lo que hace interesante correr Ford-Fulkerson en vez de
    simplemente sumar la demanda.

    Complejidad temporal: O(n + O*D), con n = filas del dataset (agregación)
    y O*D = pares origen-destino (asignación de capacidad física).
    Complejidad espacial: O(V^2) para la matriz de adyacencia, con
    V = número de nodos (orígenes + destinos + 2 nodos super).

    Args:
        df (pd.DataFrame): dataset con columnas Origen, Destino, Volumen_m3.
        capacidad_fisica (dict, opcional): {(origen, destino): capacidad_m3}.
            Si es None, se genera automáticamente con semilla fija.

    Returns:
        tuple: (grafo_capacidades, node_index, fuente, sumidero, metadatos)
            grafo_capacidades: matriz NxN de capacidades (float).
            node_index: dict que mapea 'O_<id>' / 'D_<id>' / 'SUPER_ORIGEN' /
                        'SUPER_DESTINO' al índice de fila/columna en la matriz.
            fuente / sumidero: índices de SUPER_ORIGEN y SUPER_DESTINO.
            metadatos: dict con 'demanda_od' (demanda real por par, tal cual
                       el dataset) y 'capacidad_fisica_od' (límite físico
                       simulado por par), para poder comparar ambas en los
                       reportes de main.py.
    """
    origenes = sorted(df['Origen'].unique())
    destinos = sorted(df['Destino'].unique())

    if capacidad_fisica is None:
        capacidad_fisica = generar_capacidad_fisica_infraestructura(origenes, destinos)

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

    # SUPER_ORIGEN -> cada nodo Origen: capacidad = volumen total despachado desde ese origen (DEMANDA)
    for o in origenes:
        cap = df.loc[df['Origen'] == o, 'Volumen_m3'].sum()
        grafo[node_index['SUPER_ORIGEN']][node_index[f'O_{o}']] += cap

    # Origen -> Destino: capacidad = límite FÍSICO de la carretera (independiente de la demanda)
    demanda_od = df.groupby(['Origen', 'Destino'])['Volumen_m3'].sum().to_dict()
    for o in origenes:
        for d in destinos:
            grafo[node_index[f'O_{o}']][node_index[f'D_{d}']] += capacidad_fisica[(o, d)]

    # Cada nodo Destino -> SUPER_DESTINO: capacidad = volumen total recibido en ese destino (DEMANDA)
    for d in destinos:
        cap = df.loc[df['Destino'] == d, 'Volumen_m3'].sum()
        grafo[node_index[f'D_{d}']][node_index['SUPER_DESTINO']] += cap

    fuente = node_index['SUPER_ORIGEN']
    sumidero = node_index['SUPER_DESTINO']
    metadatos = {
        "demanda_od": demanda_od,
        "capacidad_fisica_od": capacidad_fisica,
    }
    return grafo, node_index, fuente, sumidero, metadatos


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


def calcular_flujo_maximo(grafo_capacidades, fuente, sumidero, umbral_saturacion=99.99):
    """
    Inyecta tráfico por las rutas detectadas por BFS hasta que la red colapse (se llene).
    Implementación de Edmonds-Karp (Ford-Fulkerson con BFS).

    A diferencia de una versión que solo detecta si una arista llegó a 0
    (y le pega la etiqueta fija "100%"), aquí se calcula explícitamente el
    porcentaje real de saturación de CADA arista con capacidad > 0:

        porcentaje = (flujo_usado / capacidad_original) * 100

    Esto permite reportar no solo los cuellos de botella (aristas al 100%,
    o al umbral_saturacion definido) sino también el nivel de uso de las
    demás rutas, útil para la gerencia como alerta temprana de tramos que
    están cerca de saturarse aunque todavía no lo estén.

    Complejidad temporal: O(V * E^2), cota clásica de Edmonds-Karp
    (V = nodos, E = aristas de la red).
    Complejidad espacial: O(V^2) por la matriz residual.

    Args:
        grafo_capacidades (list[list[float]]): matriz de capacidades original.
        fuente (int): índice del nodo fuente.
        sumidero (int): índice del nodo sumidero.
        umbral_saturacion (float): porcentaje mínimo (0-100) para que una
            arista se considere "cuello de botella". Por defecto 99.99 para
            tolerar errores de punto flotante al comparar con 100.0 exacto.

    Returns:
        tuple: (flujo_maximo_total, cuellos_de_botella, reporte_utilizacion)
            cuellos_de_botella: lista de dicts con las aristas cuyo
                porcentaje_saturacion >= umbral_saturacion.
            reporte_utilizacion: lista de dicts con TODAS las aristas de
                capacidad > 0, cada una con su flujo_usado, capacidad_original
                y porcentaje_saturacion real (no asumido).
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

    # 3. Calculamos el % de saturación REAL de cada arista con capacidad > 0
    reporte_utilizacion = []
    for u in range(nodos):
        for v in range(nodos):
            capacidad_original = grafo_capacidades[u][v]
            if capacidad_original > 0:
                capacidad_restante = grafo_residual[u][v]
                flujo_usado = capacidad_original - capacidad_restante
                porcentaje = (flujo_usado / capacidad_original) * 100
                reporte_utilizacion.append({
                    "origen": u,
                    "destino": v,
                    "flujo_usado": round(flujo_usado, 2),
                    "capacidad_original": round(capacidad_original, 2),
                    "porcentaje_saturacion": round(porcentaje, 2),
                })

    # 4. Los cuellos de botella son solo las aristas que superan el umbral
    cuellos_de_botella = [
        item for item in reporte_utilizacion
        if item["porcentaje_saturacion"] >= umbral_saturacion
    ]

    return flujo_maximo_total, cuellos_de_botella, reporte_utilizacion


def etiquetas_cuellos_de_botella(cuellos_de_botella, node_index):
    """
    Traduce las aristas (dicts) devueltas por calcular_flujo_maximo a
    etiquetas legibles, mostrando el porcentaje REAL calculado y el
    flujo usado vs. la capacidad original (p. ej. 'O_2 -> D_7 (100.0%,
    482.3/482.3 m3)').
    """
    indice_a_nombre = {v: k for k, v in node_index.items()}
    etiquetas = []
    for item in cuellos_de_botella:
        nombre_u = indice_a_nombre[item["origen"]]
        nombre_v = indice_a_nombre[item["destino"]]
        etiquetas.append(
            f"{nombre_u} -> {nombre_v} "
            f"({item['porcentaje_saturacion']}% saturado, "
            f"{item['flujo_usado']}/{item['capacidad_original']} m3)"
        )
    return etiquetas


def top_aristas_por_utilizacion(reporte_utilizacion, node_index, top_n=5):
    """
    Devuelve las top_n aristas con mayor porcentaje de utilización, sin
    importar si llegaron o no al 100%. Útil para mostrarle a la gerencia
    qué rutas están "casi saturadas" y merecen inversión preventiva, no
    solo las que ya colapsaron.

    Complejidad temporal: O(E log E) por el ordenamiento de las aristas.
    """
    indice_a_nombre = {v: k for k, v in node_index.items()}
    ordenado = sorted(
        reporte_utilizacion, key=lambda item: item["porcentaje_saturacion"], reverse=True
    )
    resultado = []
    for item in ordenado[:top_n]:
        nombre_u = indice_a_nombre[item["origen"]]
        nombre_v = indice_a_nombre[item["destino"]]
        resultado.append(
            f"{nombre_u} -> {nombre_v}: {item['porcentaje_saturacion']}% "
            f"({item['flujo_usado']}/{item['capacidad_original']} m3)"
        )
    return resultado