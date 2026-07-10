"""
MÓDULO 2: RETO 1 - ÁRBOL AVL (INVENTARIOS Y BÚSQUEDA)
========================================================
Estructura de datos implementada desde cero (sin librerías externas) para
garantizar búsquedas O(log n) en el peor caso, incluso ante inserciones
masivas y desordenadas de ID_Producto.

Nota: Usamos 'class' únicamente como un contenedor para guardar los datos
del producto y las flechas (punteros) hacia la izquierda y derecha.
"""


class NodoAVL:
    """Representa una sola 'caja' o producto dentro del árbol."""

    def __init__(self, id_producto):
        self.id_producto = id_producto
        self.izq = None   # Puntero al hijo menor
        self.der = None   # Puntero al hijo mayor
        self.altura = 1   # Por defecto, un nodo nuevo entra en el nivel 1 (hoja)


class ArbolAVL:
    """
    Estructura que gestiona la inserción y el auto-balanceo de los nodos.

    Complejidad temporal (insertar, in_order): O(log n) por operación gracias
    al balanceo automático; O(n) para recorrer todo el árbol en in_order.
    Complejidad espacial: O(n) por los n nodos almacenados, más O(log n)
    de pila de recursión.
    """

    def obtener_altura(self, nodo):
        if not nodo:
            return 0
        return nodo.altura

    def obtener_balance(self, nodo):
        """Calcula el Factor de Equilibrio (FE = Altura_Izq - Altura_Der)."""
        if not nodo:
            return 0
        return self.obtener_altura(nodo.izq) - self.obtener_altura(nodo.der)

    # --- ROTACIONES (O(1) cada una) ---
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
        return y  # 'y' es el nuevo padre

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
        return y  # 'y' es el nuevo padre

    def insertar(self, nodo, id_producto):
        """Función recursiva para agregar un dato y rebalancear si es necesario."""
        # 1. Inserción normal de un árbol binario
        if not nodo:
            return NodoAVL(id_producto)
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

        return nodo  # Si está balanceado, no hace nada

    def in_order(self, nodo, lista_resultado):
        """Recorre el árbol priorizando la izquierda para sacar los números de menor a mayor."""
        if nodo:
            self.in_order(nodo.izq, lista_resultado)
            lista_resultado.append(nodo.id_producto)
            self.in_order(nodo.der, lista_resultado)
        return lista_resultado