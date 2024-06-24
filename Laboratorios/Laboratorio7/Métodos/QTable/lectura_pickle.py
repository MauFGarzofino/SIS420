import os
import pickle

# Construir la ruta completa al archivo pickle
def cargar_datos(metodo, tamano_mapa):
    """
    Carga datos de un archivo pickle ubicado en la estructura de directorios del proyecto.

    Parámetros:
    metodo (str): El nombre del método que queremos usar, por ejemplo, 'Acción_Valor' o 'UCB'.
    tamano_mapa (str): El tamaño del mapa, por ejemplo, '4x4' o '8x8'.

    Returns:
    data: Los datos cargados del archivo pickle.
    """
    # Obtener la ruta del script actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir la ruta completa al archivo pickle basado en el método y el tamaño del mapa
    file_path = os.path.join(current_dir, metodo, f'frozen_lake{tamano_mapa}.pkl')

    # Cargar el archivo
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    return data

# Ejemplo de uso
metodo = 'Valores_Optimistas'  # Cambiar por el método deseado, por ejemplo, 'UCB'
tamano_mapa = '4x4'      # Cambiar por el tamaño del mapa deseado, por ejemplo, '8x8' 
# (8X8) Actualmente solo disponible en el método Incremental

# Cargar y imprimir los datos
data = cargar_datos(metodo, tamano_mapa)
print(data)
