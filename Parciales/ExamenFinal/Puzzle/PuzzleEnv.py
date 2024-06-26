import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import pickle

"""
# Documentación del Modelo de Aprendizaje por Refuerzo para un Rompecabezas de 2x2

## Espacio de Observaciones
El espacio de observaciones se define como un `Box` de 2x2 con valores enteros que representan la configuración del rompecabezas. Cada estado es una configuración única de las piezas del rompecabezas.

## Espacio de Acciones
Las acciones posibles son mover la pieza vacía (representada por 0) en cuatro direcciones:
- 0: Arriba
- 1: Abajo
- 2: Izquierda
- 3: Derecha

## Esquema de Recompensas
El esquema de recompensas es el siguiente:
- Recompensa de 1 al resolver el rompecabezas (es decir, alcanzar el estado objetivo).
- Penalización de -0.1 por cada paso tomado, incentivando al agente a resolver el rompecabezas en el menor número de pasos posible.

## Exploración y Explotación
Se utiliza una política ε-greedy para equilibrar la exploración y explotación:
- `epsilon` comienza en 1, lo que significa que el agente elige acciones aleatorias inicialmente.
- `epsilon` se reduce gradualmente en cada episodio hasta alcanzar un mínimo de 0.1 (`epsilon_min`), lo que permite al agente explotar el conocimiento adquirido al elegir las mejores acciones basadas en la tabla Q.

## Planteamiento del Ejercicio
1. **Definición del Entorno:** Se define un entorno de rompecabezas de 2x2 utilizando la librería Gymnasium.
2. **Estados y Acciones:** Se representan los estados como configuraciones del rompecabezas y las acciones como movimientos posibles de la pieza vacía.
3. **Esquema de Recompensas:** Se penaliza por cada movimiento y se recompensa al resolver el rompecabezas.
4. **Política de Exploración y Explotación:** Se utiliza una política ε-greedy para seleccionar acciones.
5. **Entrenamiento del Agente:** Se entrena el agente utilizando Q-learning, actualizando la tabla Q en cada paso y reduciendo gradualmente `epsilon`.

"""

class PuzzleEnv(gym.Env):
    def __init__(self):
        super(PuzzleEnv, self).__init__()
        self.rows = 2
        self.cols = 2
        self.state = np.arange(1, self.rows * self.cols + 1).reshape((self.rows, self.cols))
        self.state[-1, -1] = 0  # Espacio vacío
        self.shuffle_state()
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=self.rows * self.cols - 1, shape=(self.rows, self.cols), dtype=int)

    def shuffle_state(self):
        flat_state = self.state.flatten()
        np.random.shuffle(flat_state)
        self.state = flat_state.reshape(self.rows, self.cols)

    def step(self, action):
        empty_pos = np.argwhere(self.state == 0)[0]
        new_pos = empty_pos.copy()
        
        if action == 0:  # up
            new_pos[0] -= 1
        elif action == 1:  # down
            new_pos[0] += 1
        elif action == 2:  # left
            new_pos[1] -= 1
        elif action == 3:  # right
            new_pos[1] += 1
        
        if (new_pos[0] >= 0 and new_pos[0] < self.rows and
            new_pos[1] >= 0 and new_pos[1] < self.cols):
            self.state[empty_pos[0], empty_pos[1]] = self.state[new_pos[0], new_pos[1]]
            self.state[new_pos[0], new_pos[1]] = 0
        
        done = np.array_equal(self.state, np.arange(1, self.rows * self.cols + 1).reshape((self.rows, self.cols)))
        reward = 1 if done else -0.1  # Penalizar cada paso y recompensar cuando se resuelva
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.arange(1, self.rows * self.cols + 1).reshape((self.rows, self.cols))
        self.state[-1, -1] = 0
        self.shuffle_state()
        return self.state

    def render(self, mode='human'):
        print(self.state)

def to_index(state):
    return tuple(state.flatten())

def choose_action(state, q_table, epsilon, action_space):
    state_idx = to_index(state)
    if state_idx not in q_table:
        q_table[state_idx] = np.zeros(action_space.n)
    if random.uniform(0, 1) < epsilon:
        return action_space.sample()
    else:
        return np.argmax(q_table[state_idx])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma, env):
    state_idx = to_index(state)
    next_state_idx = to_index(next_state)
    if next_state_idx not in q_table:
        q_table[next_state_idx] = np.zeros(env.action_space.n)
    best_next_action = np.argmax(q_table[next_state_idx])
    td_target = reward + gamma * q_table[next_state_idx][best_next_action]
    td_error = td_target - q_table[state_idx][action]
    q_table[state_idx][action] += alpha * td_error

def run(episodes, is_training=True):
    env = PuzzleEnv()
    q_table = {}
    
    epsilon = 1  # Probabilidad inicial de tomar acciones aleatorias
    epsilon_min = 0.1  # Valor mínimo de epsilon
    epsilon_decay = 0.01  # Decaimiento de epsilon
    alpha = 0.1  # Tasa de aprendizaje
    gamma = 0.99  # Factor de descuento

    for i in range(episodes):
        state = env.reset()
        done = False
        print(f"Episode {i+1}/{episodes}")

        while not done:
            action = choose_action(state, q_table, epsilon, env.action_space)
            next_state, reward, done, _ = env.step(action)
            
            if is_training:
                # Actualización de Q-learning Incremental
                update_q_table(q_table, state, action, reward, next_state, alpha, gamma, env)
            
            state = next_state

        if is_training:
            epsilon = max(epsilon * (1 - epsilon_decay), epsilon_min)
        
        # Imprimir el estado actual y la tabla Q después de cada episodio
        env.render()
        print_q_table(q_table)
            
    env.close()
    
    if is_training:
        with open("puzzle_q_table.pkl", "wb") as f:
            pickle.dump(q_table, f)

def print_q_table(q_table):
    for state, actions in q_table.items():
        print(f"State: {state}")
        print(f"Actions: {actions}")
    print("\n")

if __name__ == '__main__':
    run(100, is_training=True)
