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

from datos import generar_dataset_logistico
from arbol_avl import ArbolAVL
from knapsack import optimizar_carga_knapsack
from ford import (
    construir_grafo_desde_df,
    calcular_flujo_maximo,
    etiquetas_cuellos_de_botella,
    top_aristas_por_utilizacion,
)
from astar import buscar_ruta_a_estrella


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
    print(f" -> Combinación exacta de productos seleccionados ({len(ids_cargados)} IDs): {ids_cargados}")

    # 3. RETO FORD-FULKERSON (ahora construido desde el dataset real, con
    #    capacidad física de carretera simulada e independiente de la demanda)
    print("\n" + "-" * 70)
    print(" RETO 3: TRÁFICO MÁXIMO EN RED (FORD-FULKERSON / EDMONDS-KARP) ")
    print("-" * 70)
    red_logistica, node_index, fuente, sumidero, metadatos = construir_grafo_desde_df(df_logistica)
    flujo_max, cuellos, reporte_utilizacion = calcular_flujo_maximo(red_logistica, fuente, sumidero)
    print(f" -> Grafo construido desde df_logistica: {len(node_index)} nodos "
          f"({df_logistica['Origen'].nunique()} orígenes, {df_logistica['Destino'].nunique()} destinos).")
    print(f" -> Capacidad máxima de la red logística: {round(flujo_max, 2)} m3 (volumen equivalente).")

    print(f"\n -> Cuellos de botella reales (saturación >= 99.99%, calculada, no asumida):")
    etiquetas = etiquetas_cuellos_de_botella(cuellos, node_index)
    if etiquetas:
        for etiqueta in etiquetas:
            print(f"    * {etiqueta}")
    else:
        print("    * Ninguna arista alcanzó saturación total con este dataset.")

    print(f"\n -> Top 5 aristas con mayor % de utilización (saturadas o no):")
    for linea in top_aristas_por_utilizacion(reporte_utilizacion, node_index, top_n=5):
        print(f"    * {linea}")

    print(f"\n -> Rutas donde la DEMANDA supera la CAPACIDAD FÍSICA simulada "
          f"(inversión prioritaria):")
    demanda_od = metadatos["demanda_od"]
    capacidad_od = metadatos["capacidad_fisica_od"]
    hay_deficit = False
    for (o, d), demanda in sorted(demanda_od.items()):
        capacidad = capacidad_od[(o, d)]
        if demanda > capacidad:
            hay_deficit = True
            deficit = round(demanda - capacidad, 2)
            print(f"    * O_{o} -> D_{d}: demanda {round(demanda, 2)} m3 "
                  f"> capacidad vial {capacidad} m3 (déficit de {deficit} m3)")
    if not hay_deficit:
        print("    * Ninguna ruta tiene déficit; la infraestructura simulada alcanza para toda la demanda.")

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