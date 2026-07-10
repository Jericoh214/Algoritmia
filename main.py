import pandas as pd
import numpy as np
import heapq  # Librería nativa para manejar colas de prioridad (min-heap) en el A*

# ==============================================================================
# MÓDULO 1: GENERACIÓN DE DATOS (DATASET ESTOCÁSTICO)
# ==============================================================================
def generar_dataset_logistico():
    """
    Genera un DataFrame simulado para SafeRoute Logistics.
    Se usa una semilla (seed) para que los datos aleatorios sean siempre los 
    mismos y la auditoría pueda reproducir el experimento sin errores.
    """
    np.random.seed(42)
    n_registros = 1200 # Requisito de la rúbrica: N > 1000
    
    # Construcción de las columnas usando distribuciones uniformes y aleatorias
    data = {
        "ID_Producto": np.arange(1001, 1001 + n_registros), # Llaves primarias únicas
        "Volumen_m3": np.round(np.random.uniform(0.5, 15.0, n_registros), 2), # Rango de capacidad
        "Utilidad_USD": np.round(np.random.uniform(100.0, 5000.0, n_registros), 2), # Ganancia
        "Origen": np.random.randint(1, 5, n_registros),  # Nodos del 1 al 4 (Acopio)
        "Destino": np.random.randint(6, 10, n_registros) # Nodos del 6 al 9 (Llegada)
    }
    return pd.DataFrame(data)


# ==============================================================================
# MÓDULO 2: RETO 1 - ÁRBOL AVL (INVENTARIOS Y BÚSQUEDA)
# ==============================================================================
# Nota: Usamos 'class' únicamente como un contenedor para guardar los datos 
# del producto y las flechas (punteros) hacia la izquierda y derecha.

class NodoAVL:
    """Representa una sola 'caja' o producto dentro del árbol."""
    def __init__(self, id_producto):
        self.id_producto = id_producto
        self.izq = None  # Puntero al hijo menor
        self.der = None  # Puntero al hijo mayor
        self.altura = 1  # Por defecto, un nodo nuevo entra en el nivel 1 (hoja)

class ArbolAVL:
    """Estructura que gestiona la inserción y el auto-balanceo de los nodos."""
    
    def obtener_altura(self, nodo):
        if not nodo: return 0
        return nodo.altura

    def obtener_balance(self, nodo):
        """Calcula el Factor de Equilibrio (FE = Altura_Izq - Altura_Der)."""
        if not nodo: return 0
        return self.obtener_altura(nodo.izq) - self.obtener_altura(nodo.der)

    # --- ROTACIONES ---
    def rotacion_derecha(self, z):
        """Gira hacia la derecha cuando el peso del árbol está muy a la izquierda."""
        y = z.izq
        T3 = y.der
        # Realizamos el giro
        y.der = z
        z.izq = T3
        # Actualizamos las alturas tras el giro
        z.altura = 1 + max(self.obtener_altura(z.izq), self.obtener_altura(z.der))
        y.altura = 1 + max(self.obtener_altura(y.izq), self.obtener_altura(y.der))
        return y # 'y' es el nuevo padre

    def rotacion_izquierda(self, z):
        """Gira hacia la izquierda cuando el peso del árbol está muy a la derecha."""
        y = z.der
        T2 = y.izq
        # Realizamos el giro
        y.izq = z
        z.der = T2
        # Actualizamos las alturas
        z.altura = 1 + max(self.obtener_altura(z.izq), self.obtener_altura(z.der))
        y.altura = 1 + max(self.obtener_altura(y.izq), self.obtener_altura(y.der))
        return y # 'y' es el nuevo padre

    def insertar(self, nodo, id_producto):
        """Función recursiva para agregar un dato y rebalancear si es necesario."""
        # 1. Inserción normal de un árbol binario
        if not nodo: return NodoAVL(id_producto)
        elif id_producto < nodo.id_producto:
            nodo.izq = self.insertar(nodo.izq, id_producto)
        else:
            nodo.der = self.insertar(nodo.der, id_producto)

        # 2. Actualizamos la altura del nodo actual
        nodo.altura = 1 + max(self.obtener_altura(nodo.izq), self.obtener_altura(nodo.der))
        
        # 3. Revisamos si se rompió el balance (FE no está entre -1 y 1)
        balance = self.obtener_balance(nodo)

        # 4. Evaluamos los 4 casos de desequilibrio:
        # Caso Simple Izquierda (Requiere rotación derecha)
        if balance > 1 and id_producto < nodo.izq.id_producto:
            return self.rotacion_derecha(nodo)
        # Caso Simple Derecha (Requiere rotación izquierda)
        if balance < -1 and id_producto > nodo.der.id_producto:
            return self.rotacion_izquierda(nodo)
        # Caso Doble Izquierda-Derecha (Zig-zag izquierdo)
        if balance > 1 and id_producto > nodo.izq.id_producto:
            nodo.izq = self.rotacion_izquierda(nodo.izq)
            return self.rotacion_derecha(nodo)
        # Caso Doble Derecha-Izquierda (Zig-zag derecho)
        if balance < -1 and id_producto < nodo.der.id_producto:
            nodo.der = self.rotacion_derecha(nodo.der)
            return self.rotacion_izquierda(nodo)
        
        return nodo # Si está balanceado, no hace nada

    def in_order(self, nodo, lista_resultado):
        """Recorre el árbol priorizando la izquierda para sacar los números de menor a mayor."""
        if nodo:
            self.in_order(nodo.izq, lista_resultado)
            lista_resultado.append(nodo.id_producto)
            self.in_order(nodo.der, lista_resultado)
        return lista_resultado


# ==============================================================================
# MÓDULO 3: RETO 2 - KNAPSACK 0/1 (ASIGNACIÓN ÓPTIMA CON PROG. DINÁMICA)
# ==============================================================================
def optimizar_carga_knapsack(df, capacidad_max_m3):
    """
    Llena la mochila (camión) maximizando el valor ($) sin pasar del peso máximo.
    Convertimos los volúmenes decimales a enteros multiplicando por 100.
    """
    # Preparación de datos (Pesos = Volumen, Valores = Dinero)
    pesos = (df['Volumen_m3'] * 100).astype(int).tolist()
    valores = df['Utilidad_USD'].tolist()
    ids = df['ID_Producto'].tolist()
    
    n = len(valores)
    capacidad_entera = int(capacidad_max_m3 * 100) # Ej. 50.0m3 -> 5000 unidades

    # Creación de Matriz (Memoización). Filas=Productos, Columnas=Capacidades posibles
    K = [[0.0 for _ in range(capacidad_entera + 1)] for _ in range(n + 1)]

    # Llenado de la matriz (Bottom-Up)
    for i in range(1, n + 1):
        for w in range(1, capacidad_entera + 1):
            if pesos[i-1] <= w: # Si el producto cabe en el camión...
                # ¿Qué da más dinero? ¿Meterlo o no meterlo?
                K[i][w] = max(valores[i-1] + K[i-1][w - pesos[i-1]], K[i-1][w])
            else: # Si no cabe, copiamos el valor óptimo sin este producto
                K[i][w] = K[i-1][w]

    utilidad_maxima = K[n][capacidad_entera]
    
    # Proceso de "Backtracking": Recorrer la tabla hacia atrás para saber qué cargamos
    w_actual = capacidad_entera
    productos_seleccionados = []
    volumen_ocupado = 0.0
    utilidad_restante = utilidad_maxima

    for i in range(n, 0, -1):
        if utilidad_restante <= 0: break
        # Si la celda cambió de valor respecto a la de arriba, es que LO METIMOS al camión
        if K[i][w_actual] != K[i-1][w_actual]:
            productos_seleccionados.append(ids[i-1])
            utilidad_restante -= valores[i-1]
            w_actual -= pesos[i-1]
            volumen_ocupado += (pesos[i-1] / 100.0) # Restauramos al valor real con decimales

    return np.round(utilidad_maxima, 2), productos_seleccionados, np.round(volumen_ocupado, 2)


# ==============================================================================
# MÓDULO 4: RETO 3 - FORD-FULKERSON (EDMONDS-KARP PARA FLUJO EN REDES)
# ==============================================================================
def bfs_caminos_aumentantes(grafo_residual, fuente, sumidero, padres):
    """
    Usa Búsqueda en Anchura (BFS) para encontrar el camino más corto (en saltos)
    desde el Origen (fuente) al Destino (sumidero) que aún tenga capacidad de tráfico.
    """
    visitados = [False] * len(grafo_residual)
    cola = [fuente] # La cola es la base del BFS (First In, First Out)
    visitados[fuente] = True
    
    while cola:
        u = cola.pop(0) # Sacamos el primer nodo que entró a la cola
        
        # Revisamos todos sus vecinos
        for v, capacidad_disponible in enumerate(grafo_residual[u]):
            if not visitados[v] and capacidad_disponible > 0:
                cola.append(v)
                visitados[v] = True
                padres[v] = u # Recordamos de dónde venimos para poder trazar la ruta luego
                
                if v == sumidero: # ¡Llegamos al puerto! Hay un camino válido
                    return True
    return False # Ya no hay caminos posibles hacia la meta

def calcular_flujo_maximo(grafo_capacidades, fuente, sumidero):
    """
    Inyecta tráfico por las rutas detectadas por BFS hasta que la red colapse (se llene).
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
            
        flujo_maximo_total += flujo_camino # Sumamos este flujo a la exportación total
        
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
                cuellos_de_botella.append(f"Ruta Nodo {u} -> Nodo {v}")
                
    return flujo_maximo_total, cuellos_de_botella


# ==============================================================================
# MÓDULO 5: RETO 4 - NAVEGACIÓN INTELIGENTE CON A* (A-ESTRELLA)
# ==============================================================================
def heuristica_manhattan(a, b):
    """Fórmula h(n): Calcula la distancia ortogonal en bloques (ideal para calles de ciudad)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def buscar_ruta_a_estrella(mapa_cuadricula, inicio, meta):
    """
    Busca la ruta más rápida evadiendo tráfico (obstáculos).
    Prioriza las celdas usando f(n) = g(n) + h(n).
    """
    filas, columnas = len(mapa_cuadricula), len(mapa_cuadricula[0])
    movimientos = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Derecha, Izquierda, Abajo, Arriba
    
    # Usamos heapq (min-heap) para que Python ordene automáticamente y siempre nos 
    # devuelva la celda con el MENOR costo total 'f'
    cola_prioridad = []
    heapq.heappush(cola_prioridad, (0, 0, inicio)) # (Costo_f, Costo_g, Coordenada)
    
    padres = {inicio: None}
    costo_g = {inicio: 0} # g(n): Distancia real recorrida desde el inicio
    
    while cola_prioridad:
        _, _, nodo_actual = heapq.heappop(cola_prioridad) # Extrae la celda más prometedora
        
        if nodo_actual == meta: # Llegamos al cliente
            ruta_optima = []
            paso = meta
            while paso is not None:
                ruta_optima.append(paso)
                paso = padres[paso]
            ruta_optima.reverse()
            return ruta_optima, costo_g[meta]
            
        # Explorar calles vecinas
        for dx, dy in movimientos:
            vecino = (nodo_actual[0] + dx, nodo_actual[1] + dy)
            
            # Verificar límites de la ciudad
            if 0 <= vecino[0] < filas and 0 <= vecino[1] < columnas:
                if mapa_cuadricula[vecino[0]][vecino[1]] == 1:
                    continue # Hay tráfico, la ignoramos
                    
                nuevo_costo_g = costo_g[nodo_actual] + 1 # Nos movemos una cuadra (costo = 1)
                
                # Si esta calle es nueva, o llegamos a ella más rápido por aquí:
                if vecino not in costo_g or nuevo_costo_g < costo_g[vecino]:
                    costo_g[vecino] = nuevo_costo_g
                    
                    # f(n) = Costo_real(g) + Estimación_hacia_la_meta(h)
                    prioridad_f = nuevo_costo_g + heuristica_manhattan(vecino, meta)
                    
                    # Agregamos la calle a la cola de prioridad
                    heapq.heappush(cola_prioridad, (prioridad_f, nuevo_costo_g, vecino))
                    padres[vecino] = nodo_actual
                    
    return None, float('inf') # Fallo: Ciudad totalmente bloqueada


# ==============================================================================
# EJECUCIÓN PRINCIPAL: CONSOLA DE AUDITORÍA
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
    # Desordenamos para probar el balanceo automático del AVL
    ids_desordenados = df_logistica['ID_Producto'].sample(frac=1, random_state=42).tolist()
    for id_prod in ids_desordenados:
        raiz = arbol_inventario.insertar(raiz, id_prod)
    catalogo_ordenado = arbol_inventario.in_order(raiz, [])
    print(f" -> Catálogo exportado y ordenado. Verificación (Top 5): {catalogo_ordenado[:5]}")

    # 2. RETO KNAPSACK
    print("\n" + "-" * 70)
    print(" RETO 2: ASIGNACIÓN DE RECURSOS (MOCHILA 0/1) ")
    print("-" * 70)
    cap_camion = 50.0 # 50 metros cúbicos
    ganancia, ids_cargados, espacio = optimizar_carga_knapsack(df_logistica, cap_camion)
    print(f" -> Capacidad: {cap_camion} m3 | Utilizado: {espacio} m3 | Carga: {len(ids_cargados)} items")
    print(f" -> Ganancia Neta Optimizada: ${ganancia} USD")

    # 3. RETO FORD-FULKERSON
    print("\n" + "-" * 70)
    print(" RETO 3: TRÁFICO MÁXIMO EN RED (FORD-FULKERSON) ")
    print("-" * 70)
    # Matriz 6x6. Fila 0 es el Origen, Fila 5 es el Puerto
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
    # 0 = Libre, 1 = Embudo/Tráfico
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