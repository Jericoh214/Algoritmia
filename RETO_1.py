# ==============================================================================
# RETO 1: CONSISTENCIA DE INVENTARIOS Y ESTRUCTURAS CORE (ÁRBOL AVL)
# ==============================================================================

class NodoAVL:
    """
    Clase base que representa la unidad fundamental de nuestro inventario.
    Cada nodo es un 'producto' dentro de la estructura jerárquica.
    """
    def __init__(self, id_producto):
        self.id_producto = id_producto  # La llave primaria (ej. 1001, 1002)
        self.izq = None                 # Puntero al subárbol izquierdo (IDs menores)
        self.der = None                 # Puntero al subárbol derecho (IDs mayores)
        self.altura = 1                 # Todo nodo nuevo entra como una 'hoja', por lo que su altura inicial es 1


class ArbolAVL:
    """
    Estructura principal que gestiona el catálogo. Garantiza búsquedas en O(log n)
    al mantener el árbol estrictamente balanceado tras cada inserción.
    """

    # --- 1. FUNCIONES AUXILIARES DE CÁLCULO ---
    
    def obtener_altura(self, nodo):
        """Retorna la altura actual de un nodo. Si el nodo es nulo (llegamos al final), su altura es 0."""
        if not nodo:
            return 0
        return nodo.altura

    def obtener_balance(self, nodo):
        """
        Calcula el Factor de Equilibrio (FE) restando la altura izquierda menos la derecha.
        - Si el FE es mayor a 1, el árbol pesa mucho hacia la izquierda.
        - Si el FE es menor a -1, el árbol pesa mucho hacia la derecha.
        """
        if not nodo:
            return 0
        return self.obtener_altura(nodo.izq) - self.obtener_altura(nodo.der)


    # --- 2. MOTORES DE BALANCEO (ROTACIONES) ---
    
    def rotacion_derecha(self, z):
        """
        Se ejecuta cuando el subárbol izquierdo está muy pesado (FE > 1).
        Gira los nodos hacia la derecha (como un volante) para reequilibrarlos.
        'z' es el nodo desbalanceado, 'y' será la nueva raíz de este subárbol.
        """
        y = z.izq
        T3 = y.der # T3 es el hijo derecho de 'y' que quedará huérfano temporalmente

        # Movimientos físicos de la rotación
        y.der = z
        z.izq = T3 # 'z' adopta a T3 como su nuevo hijo izquierdo

        # Actualizamos las alturas tras el movimiento (primero 'z' que ahora está abajo, luego 'y')
        z.altura = 1 + max(self.obtener_altura(z.izq), self.obtener_altura(z.der))
        y.altura = 1 + max(self.obtener_altura(y.izq), self.obtener_altura(y.der))

        return y # Retornamos 'y' para que se conecte con el resto del árbol padre

    def rotacion_izquierda(self, z):
        """
        Se ejecuta cuando el subárbol derecho está muy pesado (FE < -1).
        Gira los nodos hacia la izquierda.
        """
        y = z.der
        T2 = y.izq

        # Movimientos físicos de la rotación
        y.izq = z
        z.der = T2

        # Actualizamos alturas
        z.altura = 1 + max(self.obtener_altura(z.izq), self.obtener_altura(z.der))
        y.altura = 1 + max(self.obtener_altura(y.izq), self.obtener_altura(y.der))

        return y


    # --- 3. LÓGICA CORE: INSERCIÓN CON BALANCEO AUTOMÁTICO ---
    
    def insertar(self, nodo, id_producto):
        """
        Inserta un nuevo ID_Producto de forma recursiva y verifica si el 
        árbol se rompió para aplicar la rotación correspondiente.
        """
        # PASO 1: Inserción normal de un Árbol Binario de Búsqueda (BST)
        if not nodo:
            return NodoAVL(id_producto) # Si encontramos un espacio vacío, creamos el nodo
        elif id_producto < nodo.id_producto:
            nodo.izq = self.insertar(nodo.izq, id_producto) # Va a la izquierda si es menor
        else:
            nodo.der = self.insertar(nodo.der, id_producto) # Va a la derecha si es mayor

        # PASO 2: Actualizar la altura del nodo actual (ancestro)
        nodo.altura = 1 + max(self.obtener_altura(nodo.izq), self.obtener_altura(nodo.der))

        # PASO 3: Calcular el Factor de Equilibrio (FE) para revisar desbalances
        balance = self.obtener_balance(nodo)

        # PASO 4: Analizar los 4 casos de desbalance y aplicar rotaciones

        # Caso 1: Desbalance a la Izquierda-Izquierda (Requiere rotación simple a la derecha)
        # El FE es positivo (>1) y el nuevo ID se insertó a la izquierda del hijo izquierdo.
        if balance > 1 and id_producto < nodo.izq.id_producto:
            return self.rotacion_derecha(nodo)

        # Caso 2: Desbalance a la Derecha-Derecha (Requiere rotación simple a la izquierda)
        # El FE es negativo (<-1) y el nuevo ID se insertó a la derecha del hijo derecho.
        if balance < -1 and id_producto > nodo.der.id_producto:
            return self.rotacion_izquierda(nodo)

        # Caso 3: Desbalance Izquierda-Derecha (Requiere Rotación Doble)
        # El FE es positivo, pero el ID se insertó a la *derecha* del hijo izquierdo (forma de 'zigzag').
        if balance > 1 and id_producto > nodo.izq.id_producto:
            nodo.izq = self.rotacion_izquierda(nodo.izq) # Primera rotación en el hijo
            return self.rotacion_derecha(nodo)           # Segunda rotación en el padre

        # Caso 4: Desbalance Derecha-Izquierda (Requiere Rotación Doble)
        # El FE es negativo, pero el ID se insertó a la *izquierda* del hijo derecho.
        if balance < -1 and id_producto < nodo.der.id_producto:
            nodo.der = self.rotacion_derecha(nodo.der)   # Primera rotación en el hijo
            return self.rotacion_izquierda(nodo)         # Segunda rotación en el padre

        # Si llegamos aquí, el nodo estaba balanceado (FE entre -1 y 1). Solo lo retornamos.
        return nodo


    # --- 4. EXPORTACIÓN DE DATOS ---
    
    def in_order(self, nodo, lista_resultado):
        """
        Recorre el árbol usando la estrategia In-Order: 
        1. Subárbol Izquierdo
        2. Raíz (Se guarda el dato)
        3. Subárbol Derecho
        Esto garantiza matemáticamente que los IDs se exporten ordenados de menor a mayor.
        """
        if nodo:
            self.in_order(nodo.izq, lista_resultado)
            lista_resultado.append(nodo.id_producto)
            self.in_order(nodo.der, lista_resultado)
        return lista_resultado