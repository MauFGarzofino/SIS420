import pickle

# Ruta al archivo .pkl del método que queramos observar

# Cambiar /UCB por el método del que se quiera ver la tabla Q
#file_path = 'Laboratorios/Laboratorio7/Métodos/QTable/UCB/frozen_lake4x4.pkl'
file_path = 'Laboratorios/Laboratorio7/Métodos/QTable/Gradiente/frozen_lake4x4_H.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

print(data)
