import gymnasium as gym
import numpy as np

def value_iteration(env):
    # Umbral para verificar la convergencia de la función de valor
    theta = 1e-6
    
    # Factor de descuento
    gamma = 0.99
    
    # Inicializamos la tabla de valores y la política con ceros
    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n, dtype=int)
    
    while True:
        # Inicializamos delta para medir la diferencia máxima entre actualizaciones
        delta = 0
        
        for state in range(env.observation_space.n):
            # Guardamos el valor actual de V[state] para comparar después
            v = V[state]
            # Inicializamos los valores Q para todas las acciones posibles en el estado actual
            q_values = np.zeros(env.action_space.n)
            
            for action in range(env.action_space.n):
                # Calculamos los Q-valores para cada acción en el estado actual
                for prob, next_state, reward, terminated in env.P[state][action]:
                    q_values[action] += prob * (reward + gamma * V[next_state] * (not terminated))
            
            # Actualizamos el valor del estado con el máximo Q-valor
            V[state] = np.max(q_values)
            # Actualizamos delta con la máxima diferencia entre los valores antiguos y nuevos
            delta = max(delta, np.abs(v - V[state]))
        
        # Si delta es menor que theta, la convergencia se ha alcanzado y salimos del bucle
        if delta < theta:
            break
    
    # Actualizamos la política seleccionando la acción que maximiza el Q-valor para cada estado
    for state in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for prob, next_state, reward, terminated in env.P[state][action]:
                q_values[action] += prob * (reward + gamma * V[next_state] * (not terminated))
        policy[state] = np.argmax(q_values)
    
    return policy, V

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

def run_value_iteration(episodes=1, render=False):
    # Creación del entorno Frozen Lake con renderizado 'human' si se especifica
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human' if render else None)
    # Ejecución de la iteración de valores
    policy, V = value_iteration(env)
    
    # Imprimimos la política y la función de valor
    print("Policy:")
    print(policy.reshape((4, 4)))
    print("Value Function:")
    print(V.reshape((4, 4)))

    # Si se especifica, visualizamos la política
    if render:
        for _ in range(episodes):
            visualize_policy(env, policy)
    
    env.close()

# Entrenamos al agente y lo ponemos a prueba
if __name__ == '__main__':
    run_value_iteration(episodes=1, render=True)