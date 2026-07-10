import pandas as pd
import numpy as np
import heapq

# ==============================================================================
# MÓDULO 1: GENERACIÓN DE DATOS (DATASET ESTOCÁSTICO)
# ==============================================================================
def generar_dataset_logistico():
    """Genera una muestra de más de 1000 registros estocásticos para el proyecto."""
    np.random.seed(42)
    n_registros = 1200
    
    data = {
        "ID_Producto": np.arange(1001, 1001 + n_registros),
        "Volumen_m3": np.round(np.random.uniform(0.5, 15.0, n_registros), 2),
        "Utilidad_USD": np.round(np.random.uniform(100.0, 5000.0, n_registros), 2),
        "Origen": np.random.randint(1, 5, n_registros),
        "Destino": np.random.randint(6, 10, n_registros)
    }
    return pd.DataFrame(data)


# ==============================================================================
# MÓDULO 2: RETO 1 - ÁRBOL AVL (INVENTARIOS Y BÚSQUEDA)
# ==============================================================================
class NodoAVL:
    def __init__(self, id_producto):
        self.id_producto = id_producto
        self.izq = None
        self.der = None
        self.altura = 1

class ArbolAVL:
    def obtener_altura(self, nodo):
        if not nodo: return 0
        return nodo.altura

    def obtener_balance(self, nodo):
        if not nodo: return 0
        return self.obtener_altura(nodo.izq) - self.obtener_altura(nodo.der)

    def rotacion_derecha(self, z):
        y = z.izq
        T3 = y.der
        y.der = z
        z.izq = T3
        z.altura = 1 + max(self.obtener_altura(z.izq), self.obtener_altura(z.der))
        y.altura = 1 + max(self.obtener_altura(y.izq), self.obtener_altura(y.der))
        return y

    def rotacion_izquierda(self, z):
        y = z.der
        T2 = y.izq
        y.izq = z
        z.der = T2
        z.altura = 1 + max(self.obtener_altura(z.izq), self.obtener_altura(z.der))
        y.altura = 1 + max(self.obtener_altura(y.izq), self.obtener_altura(y.der))
        return y

    def insertar(self, nodo, id_producto):
        if not nodo: return NodoAVL(id_producto)
        elif id_producto < nodo.id_producto:
            nodo.izq = self.insertar(nodo.izq, id_producto)
        else:
            nodo.der = self.insertar(nodo.der, id_producto)

        nodo.altura = 1 + max(self.obtener_altura(nodo.izq), self.obtener_altura(nodo.der))
        balance = self.obtener_balance(nodo)

        if balance > 1 and id_producto < nodo.izq.id_producto:
            return self.rotacion_derecha(nodo)
        if balance < -1 and id_producto > nodo.der.id_producto:
            return self.rotacion_izquierda(nodo)
        if balance > 1 and id_producto > nodo.izq.id_producto:
            nodo.izq = self.rotacion_izquierda(nodo.izq)
            return self.rotacion_derecha(nodo)
        if balance < -1 and id_producto < nodo.der.id_producto:
            nodo.der = self.rotacion_derecha(nodo.der)
            return self.rotacion_izquierda(nodo)
        return nodo

    def in_order(self, nodo, lista_resultado):
        if nodo:
            self.in_order(nodo.izq, lista_resultado)
            lista_resultado.append(nodo.id_producto)
            self.in_order(nodo.der, lista_resultado)
        return lista_resultado


# ==============================================================================
# MÓDULO 3: RETO 2 - KNAPSACK 0/1 (ASIGNACIÓN ÓPTIMA)
# ==============================================================================
def optimizar_carga_knapsack(df, capacidad_max_m3):
    """Programación Dinámica (Bottom-Up) escalando decimales a enteros."""
    pesos = (df['Volumen_m3'] * 100).astype(int).tolist()
    valores = df['Utilidad_USD'].tolist()
    ids = df['ID_Producto'].tolist()
    
    n = len(valores)
    capacidad_entera = int(capacidad_max_m3 * 100)

    K = [[0.0 for _ in range(capacidad_entera + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacidad_entera + 1):
            if pesos[i-1] <= w:
                K[i][w] = max(valores[i-1] + K[i-1][w - pesos[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]

    utilidad_maxima = K[n][capacidad_entera]
    w_actual = capacidad_entera
    productos_seleccionados = []
    volumen_ocupado = 0.0
    utilidad_restante = utilidad_maxima

    for i in range(n, 0, -1):
        if utilidad_restante <= 0: break
        if K[i][w_actual] != K[i-1][w_actual]:
            productos_seleccionados.append(ids[i-1])
            utilidad_restante -= valores[i-1]
            w_actual -= pesos[i-1]
            volumen_ocupado += (pesos[i-1] / 100.0)

    return np.round(utilidad_maxima, 2), productos_seleccionados, np.round(volumen_ocupado, 2)


# ==============================================================================
# MÓDULO 4: RETO 3 - FORD-FULKERSON / EDMONDS-KARP (REDES)
# ==============================================================================
def bfs_caminos_aumentantes(grafo_residual, fuente, sumidero, padres):
    visitados = [False] * len(grafo_residual)
    cola = [fuente]
    visitados[fuente] = True
    
    while cola:
        u = cola.pop(0)
        for v, capacidad_disponible in enumerate(grafo_residual[u]):
            if not visitados[v] and capacidad_disponible > 0:
                cola.append(v)
                visitados[v] = True
                padres[v] = u
                if v == sumidero:
                    return True
    return False

def calcular_flujo_maximo(grafo_capacidades, fuente, sumidero):
    nodos = len(grafo_capacidades)
    grafo_residual = [fila[:] for fila in grafo_capacidades]
    padres = [-1] * nodos
    flujo_maximo_total = 0
    
    while bfs_caminos_aumentantes(grafo_residual, fuente, sumidero, padres):
        flujo_camino = float("Inf")
        s = sumidero
        while s != fuente:
            flujo_camino = min(flujo_camino, grafo_residual[padres[s]][s])
            s = padres[s]
            
        flujo_maximo_total += flujo_camino
        
        v = sumidero
        while v != fuente:
            u = padres[v]
            grafo_residual[u][v] -= flujo_camino
            grafo_residual[v][u] += flujo_camino
            v = padres[v]
            
    cuellos_de_botella = []
    for u in range(nodos):
        for v in range(nodos):
            if grafo_capacidades[u][v] > 0 and grafo_residual[u][v] == 0:
                cuellos_de_botella.append(f"Ruta Nodo {u} -> Nodo {v}")
                
    return flujo_maximo_total, cuellos_de_botella


# ==============================================================================
# MÓDULO 5: RETO 4 - A* ESTRELLA (NAVEGACIÓN)
# ==============================================================================
def heuristica_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def buscar_ruta_a_estrella(mapa_cuadricula, inicio, meta):
    filas, columnas = len(mapa_cuadricula), len(mapa_cuadricula[0])
    movimientos = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    cola_prioridad = []
    heapq.heappush(cola_prioridad, (0, 0, inicio))
    padres = {inicio: None}
    costo_g = {inicio: 0}
    
    while cola_prioridad:
        _, _, nodo_actual = heapq.heappop(cola_prioridad)
        
        if nodo_actual == meta:
            ruta_optima = []
            paso = meta
            while paso is not None:
                ruta_optima.append(paso)
                paso = padres[paso]
            ruta_optima.reverse()
            return ruta_optima, costo_g[meta]
            
        for dx, dy in movimientos:
            vecino = (nodo_actual[0] + dx, nodo_actual[1] + dy)
            if 0 <= vecino[0] < filas and 0 <= vecino[1] < columnas:
                if mapa_cuadricula[vecino[0]][vecino[1]] == 1:
                    continue
                nuevo_costo_g = costo_g[nodo_actual] + 1
                if vecino not in costo_g or nuevo_costo_g < costo_g[vecino]:
                    costo_g[vecino] = nuevo_costo_g
                    prioridad_f = nuevo_costo_g + heuristica_manhattan(vecino, meta)
                    heapq.heappush(cola_prioridad, (prioridad_f, nuevo_costo_g, vecino))
                    padres[vecino] = nodo_actual
                    
    return None, float('inf')


# ==============================================================================
# EJECUCIÓN PRINCIPAL Y CONSOLA CORPORATIVA
# ==============================================================================
def main():
    print("\n" + "="*70)
    print(" SISTEMA INTEGRAL DE OPTIMIZACIÓN - SAFEROUTE LOGISTICS ".center(70))
    print("="*70)
    
    # 0. Datos
    df_logistica = generar_dataset_logistico()
    print(f"\n[SISTEMA] Dataset estocástico inicializado con {len(df_logistica)} registros.\n")

    # 1. RETO AVL
    print("-" * 70)
    print(" RETO 1: CONSISTENCIA DE INVENTARIOS (ÁRBOL AVL) ")
    print("-" * 70)
    arbol_inventario = ArbolAVL()
    raiz = None
    ids_desordenados = df_logistica['ID_Producto'].sample(frac=1, random_state=42).tolist()
    for id_prod in ids_desordenados:
        raiz = arbol_inventario.insertar(raiz, id_prod)
    catalogo_ordenado = arbol_inventario.in_order(raiz, [])
    print(f" -> Catálogo exportado y ordenado. Verificación (Top 5): {catalogo_ordenado[:5]}")

    # 2. RETO KNAPSACK
    print("\n" + "-" * 70)
    print(" RETO 2: ASIGNACIÓN DE RECURSOS (MOCHILA 0/1) ")
    print("-" * 70)
    cap_camion = 50.0 
    ganancia, ids_cargados, espacio = optimizar_carga_knapsack(df_logistica, cap_camion)
    print(f" -> Capacidad: {cap_camion} m3 | Utilizado: {espacio} m3 | Carga: {len(ids_cargados)} items")
    print(f" -> Ganancia Neta Optimizada: ${ganancia} USD")

    # 3. RETO FORD-FULKERSON
    print("\n" + "-" * 70)
    print(" RETO 3: TRÁFICO MÁXIMO EN RED (FORD-FULKERSON) ")
    print("-" * 70)
    red_logistica = [
        [0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]
    ]
    flujo_max, cuellos = calcular_flujo_maximo(red_logistica, 0, 5)
    print(f" -> Capacidad máxima de la red logística: {flujo_max} toneladas/hora.")
    for cuello in cuellos: print(f"    * {cuello} - [Saturación al 100%]")

    # 4. RETO A-ESTRELLA
    print("\n" + "-" * 70)
    print(" RETO 4: RUTAS DE ÚLTIMA MILLA (A-ESTRELLA CON IA) ")
    print("-" * 70)
    mapa_urbano = [
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0]
    ]
    ruta, costo = buscar_ruta_a_estrella(mapa_urbano, (0, 0), (5, 5))
    print(f" -> Ruta evasiva trazada con éxito. Costo (Combustible): {costo} bloques.")
    print(f" -> Trayectoria (Coordenadas): {ruta}")
    
    print("\n" + "="*70)
    print(" EJECUCIÓN FINALIZADA. LISTO PARA AUDITORÍA. ".center(70))
    print("="*70 + "\n")

if __name__ == "__main__":
    main()