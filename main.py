#Comentar 

"""
EJECUCIÓN PRINCIPAL: CONSOLA DE AUDITORÍA
=============================================
Orquesta los 4 retos usando los módulos separados. No contiene lógica de
algoritmos: solo importa, ejecuta y reporta resultados.

Además de imprimir por consola, genera automáticamente el archivo
'evidencia_ejecucion.txt' que exige el PDF de la EC3 como entregable
obligatorio ("Evidencias de Ejecución").
"""

import sys
import io

from datos import generar_dataset_logistico #importamos el dataset (datos)
from arbol_avl import ArbolAVL #importamos la clase arbol AVL (Funcionnamiento del arbol autobalanceado)
from knapsack import optimizar_carga_knapsack #mochila (halla combinacion de productos que maximiza la utilidad neta sin exceder la capacidad de carga del camión (Volumen_m3).)
from ford import (  #Importamos funciones del archivo ford.py
    construir_grafo_desde_df,   
    calcular_flujo_maximo, 
    etiquetas_cuellos_de_botella,
) 
from astar import buscar_ruta_a_estrella  # A* (busca la ruta optima en un camino(0) lleno de obstaculos (1)


class Tee:
    """Escribe simultáneamente a la consola y a un buffer de texto,
    para poder mostrar el progreso en pantalla y además dejar constancia
    en el archivo de evidencia sin duplicar los print()."""

    def __init__(self, *destinos):
        self.destinos = destinos

    def write(self, texto):
        for d in self.destinos:
            d.write(texto)

    def flush(self):
        for d in self.destinos:
            d.flush()


def main():
    print("\n" + "=" * 70)
    print(" SISTEMA INTEGRAL DE OPTIMIZACIÓN - SAFEROUTE LOGISTICS ".center(70))
    print("=" * 70) 

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
    print(f" -> Total de productos indexados: {len(catalogo_ordenado)}")

    # 2. RETO KNAPSACK
    print("\n" + "-" * 70)
    print(" RETO 2: ASIGNACIÓN DE RECURSOS (MOCHILA 0/1) ")
    print("-" * 70)
    cap_camion = 50.0  # 50 metros cúbicos
    ganancia, ids_cargados, espacio = optimizar_carga_knapsack(df_logistica, cap_camion)
    print(f" -> Capacidad: {cap_camion} m3 | Utilizado: {espacio} m3 | Carga: {len(ids_cargados)} items")
    print(f" -> Ganancia Neta Optimizada: ${ganancia} USD")
    print(f" -> IDs seleccionados (primeros 10): {ids_cargados[:10]}")

    # 3. RETO FORD-FULKERSON (ahora construido desde el dataset real)
    print("\n" + "-" * 70)
    print(" RETO 3: TRÁFICO MÁXIMO EN RED (FORD-FULKERSON / EDMONDS-KARP) ")
    print("-" * 70)
    red_logistica, node_index, fuente, sumidero = construir_grafo_desde_df(df_logistica)
    flujo_max, cuellos = calcular_flujo_maximo(red_logistica, fuente, sumidero)
    print(f" -> Grafo construido desde df_logistica: {len(node_index)} nodos "
          f"({df_logistica['Origen'].nunique()} orígenes, {df_logistica['Destino'].nunique()} destinos).")
    print(f" -> Capacidad máxima de la red logística: {round(flujo_max, 2)} m3 (volumen equivalente).")
    for etiqueta in etiquetas_cuellos_de_botella(cuellos, node_index):
        print(f"    * {etiqueta} - [Saturación al 100%]")

    # 4. RETO A-ESTRELLA
    print("\n" + "-" * 70)
    print(" RETO 4: RUTAS DE ÚLTIMA MILLA (A-ESTRELLA CON IA) ")
    print("-" * 70)
    # 0 = Libre, 1 = Embudo/Tráfico
    mapa_urbano = [
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
    ]
    ruta, costo = buscar_ruta_a_estrella(mapa_urbano, (0, 0), (5, 5))
    print(f" -> Ruta evasiva trazada con éxito. Costo (Combustible): {costo} bloques.")
    print(f" -> Trayectoria (Coordenadas): {ruta}")

    print("\n" + "=" * 70)
    print(" EJECUCIÓN FINALIZADA. LISTO PARA AUDITORÍA. ".center(70))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Capturamos toda la salida en un buffer, la mostramos por consola en
    # tiempo real (Tee) y al final la volcamos a evidencia_ejecucion.txt,
    # tal como exige el PDF de la EC3.
    buffer = io.StringIO()
    stdout_original = sys.stdout
    sys.stdout = Tee(stdout_original, buffer)
    try:
        main()
    finally:
        sys.stdout = stdout_original

    with open("evidencia_ejecucion.txt", "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    print("[SISTEMA] Evidencia guardada en 'evidencia_ejecucion.txt'")