import pandas as pd

import numpy as np

 

# Configurar semilla para reproducibilidad de la auditoría técnica

np.random.seed(42)

 

# Simulación de N > 1000 registros

n_registros = 1200

 

data = {

    "ID_Producto": np.arange(1001, 1001 + n_registros),

    "Volumen_m3": np.round(np.random.uniform(0.5, 15.0, n_registros), 2),

    "Utilidad_USD": np.round(np.random.uniform(100.0, 5000.0, n_registros), 2),

    "Origen": np.random.randint(1, 5, n_registros),      # Centros de acopio iniciales

    "Destino": np.random.randint(6, 10, n_registros)     # Puertos finales de llegada

}

 

df_logistica = pd.DataFrame(data)

print(f"Dataset generado exitosamente. Total de filas: {len(df_logistica)}")

 