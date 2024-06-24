import numpy as np
import gymnasium as gym

# is_slippery = False, cada acción tiene un resultado predecible, lo que simplifica la implementación de la ecuación de Bellman.
env = gym.make('FrozenLake-v1', is_slippery=False)
gamma = 0.99  # Factor de descuento
theta = 1e-8  # Umbral de convergencia

def value_iteration(env, gamma, theta):
    # Inicializamos la tabla de valores con el valor de todos los estados a cero
    V = np.zeros(env.observation_space.n)
    
    while True:
        delta = 0
        # Para cada estado en el espacio de observación
        for s in range(env.observation_space.n):
            v = V[s]
            # Calcular el valor máximo para todas las acciones posibles en el estado s
            V[s] = max([sum([p * (r + gamma * V[s_])
                             for p, s_, r, _ in env.P[s][a]])
                        for a in range(env.action_space.n)])
            # Actualizamos delta con la máxima diferencia entre los valores antiguos y nuevos
            delta = max(delta, abs(v - V[s]))
        # Si delta es menor que theta, la convergencia se ha alcanzado y salimos del bucle
        if delta < theta:
            break
    
    # Derivar la política óptima de los valores de estado
    policy = np.zeros(env.observation_space.n, dtype=int)
    for s in range(env.observation_space.n):
        # Seleccionar la acción que maximiza el valor esperado
        policy[s] = np.argmax([sum([p * (r + gamma * V[s_])
                                    for p, s_, r, _ in env.P[s][a]])
                               for a in range(env.action_space.n)])
    
    return policy, V

def print_policy_and_value(policy, V):
    # Imprimir la política en formato 4x4
    print("Policy:")
    print(policy.reshape((4, 4)))
    
    # Imprimir la función de valor en formato 4x4
    print("Value Function:")
    print(V.reshape((4, 4)))

def visualize_policy(env, policy):
    # Inicializamos el estado del entorno
    state = env.reset()[0]
    env.render()
    terminated = False
    truncated = False
    # Seguimos la política óptima hasta que el episodio termine
    while not terminated and not truncated:
        action = policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        env.render()

def run_value_iteration_and_record(episodes=1, render=False):
    # Creación del entorno Frozen Lake con renderizado 'human' si se especifica
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human' if render else None)
    # Ejecución de la iteración de valores
    policy, V = value_iteration(env, gamma, theta)
    
    # Imprimimos la política y la función de valor
    print_policy_and_value(policy, V)
    
    # Si se especifica, visualizamos la política
    if render:
        for _ in range(episodes):
            visualize_policy(env, policy)
    
    env.close()

# Entrenamos al agente y lo ponemos a prueba
if __name__ == '__main__':
    run_value_iteration_and_record(episodes=1, render=True)