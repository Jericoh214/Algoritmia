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
import pandas as pd

from datos import generar_dataset_logistico
from arbol_avl import ArbolAVL
from knapsack import optimizar_carga_knapsack
from ford import construir_grafo_desde_df, calcular_flujo_maximo
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
    print(f" -> Catálogo exportado y ordenado. Total de productos indexados: {len(catalogo_ordenado)}\n")

    ids_muestra = catalogo_ordenado[:5] + catalogo_ordenado[-5:]
    tabla_catalogo = df_logistica[df_logistica['ID_Producto'].isin(ids_muestra)].sort_values('ID_Producto')
    print(" -> Verificación (Top 5 + Bottom 5 del recorrido In-Order):")
    print(tabla_catalogo.to_string(index=False))

    # 2. RETO KNAPSACK
    print("\n" + "-" * 70)
    print(" RETO 2: ASIGNACIÓN DE RECURSOS (MOCHILA 0/1) ")
    print("-" * 70)
    cap_camion = 50.0  # 50 metros cúbicos
    ganancia, ids_cargados, espacio = optimizar_carga_knapsack(df_logistica, cap_camion)
    print(f" -> Capacidad: {cap_camion} m3 | Utilizado: {espacio} m3 | Carga: {len(ids_cargados)} items")
    print(f" -> Ganancia Neta Optimizada: ${ganancia} USD\n")

    tabla_seleccion = df_logistica[df_logistica['ID_Producto'].isin(ids_cargados)].sort_values('ID_Producto')
    print(f" -> Combinación exacta de productos seleccionados ({len(ids_cargados)} items):")
    print(tabla_seleccion.to_string(index=False))
    print(f"\n -> Verificación: suma Volumen_m3 de la tabla = "
          f"{round(tabla_seleccion['Volumen_m3'].sum(), 2)} m3 (debe coincidir con 'Utilizado')")
    print(f" -> Verificación: suma Utilidad_USD de la tabla = "
          f"${round(tabla_seleccion['Utilidad_USD'].sum(), 2)} (debe coincidir con 'Ganancia Neta Optimizada')")

    # 3. RETO FORD-FULKERSON (ahora construido desde el dataset real, con
    #    capacidad física de carretera simulada e independiente de la demanda)
    print("\n" + "-" * 70)
    print(" RETO 3: TRÁFICO MÁXIMO EN RED (FORD-FULKERSON / EDMONDS-KARP) ")
    print("-" * 70)
    red_logistica, node_index, fuente, sumidero, metadatos = construir_grafo_desde_df(df_logistica)
    flujo_max, cuellos, reporte_utilizacion = calcular_flujo_maximo(red_logistica, fuente, sumidero)
    print(f" -> Grafo construido desde df_logistica: {len(node_index)} nodos "
          f"({df_logistica['Origen'].nunique()} orígenes, {df_logistica['Destino'].nunique()} destinos).")
    print(f" -> Capacidad máxima de la red logística: {round(flujo_max, 2)} m3 (volumen equivalente).\n")

    # Tabla consolidada por cada arista Origen -> Destino (columnas Origen y
    # Destino, tal como existen en df_logistica), comparando demanda real del
    # dataset contra la capacidad física simulada de la carretera y el % de
    # saturación calculado por Edmonds-Karp.
    indice_a_nombre = {v: k for k, v in node_index.items()}
    demanda_od = metadatos["demanda_od"]
    capacidad_od = metadatos["capacidad_fisica_od"]
    filas_red = []
    for item in reporte_utilizacion:
        nombre_u = indice_a_nombre[item["origen"]]
        nombre_v = indice_a_nombre[item["destino"]]
        if nombre_u.startswith("O_") and nombre_v.startswith("D_"):
            o, d = int(nombre_u.split("_")[1]), int(nombre_v.split("_")[1])
            demanda = round(demanda_od.get((o, d), 0.0), 2)
            capacidad = capacidad_od[(o, d)]
            filas_red.append({
                "Origen": o,
                "Destino": d,
                "Demanda_m3": demanda,
                "Capacidad_Vial_m3": capacidad,
                "Flujo_Usado_m3": item["flujo_usado"],
                "Saturacion_%": item["porcentaje_saturacion"],
                "Cuello_Botella": "SI" if item["porcentaje_saturacion"] >= 99.99 else "no",
                "Deficit_m3": round(max(demanda - capacidad, 0.0), 2),
            })
    tabla_red = pd.DataFrame(filas_red).sort_values(["Origen", "Destino"])
    print(" -> Detalle por ruta Origen -> Destino:")
    print(tabla_red.to_string(index=False))
    print(f"\n -> Total de rutas en cuello de botella (saturación >= 99.99%): "
          f"{(tabla_red['Cuello_Botella'] == 'SI').sum()} de {len(tabla_red)}")
    print(f" -> Total de rutas con déficit de infraestructura (demanda > capacidad vial): "
          f"{(tabla_red['Deficit_m3'] > 0).sum()} de {len(tabla_red)}")

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