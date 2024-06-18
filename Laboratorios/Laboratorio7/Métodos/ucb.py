import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

# Register the enhanced frozen lake environment
gym.register(
    id="FrozenLake-enhanced",
    entry_point="frozen_lake_enhanced:FrozenLakeEnv",
    kwargs={"map_name": "4x4"},
    max_episode_steps=200,
    reward_threshold=0.85,
)

def run(episodes, is_training=True, render=False):
    # 'FrozenLake-enhanced' es el id especificado anteriormente
    env = gym.make('FrozenLake-enhanced', desc=None, map_name="4x4", is_slippery=False, render_mode='human' if render else None)
    
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Inicializar la tabla Q
        acciones = np.ones((env.observation_space.n, env.action_space.n))  # Contador de acciones tomadas (iniciadas en 1)
    else:
        with open('Laboratorios/Laboratorio7/Métodos/QTable/UCB/frozen_lake4x4.pkl', 'rb') as f:
            q = pickle.load(f)
            acciones = np.ones((env.observation_space.n, env.action_space.n))  # Se puede cargar también si se ha guardado anteriormente
        
    use_ucb = True  # Controla si se usa UCB
    epsilon = 0.1  # Probabilidad de tomar acciones aleatorias en \(\epsilon\)-greedy
    epsilon_min = 0 # Épsilon mínimo
    epsilon_decay = 0.001  # Tasa de disminución de epsilon 
    alpha = 0.5  # Tasa de aprendizaje
    c = 0.8  # Constante para UCB
    rng = np.random.default_rng()  # Generador de números aleatorios

    rewards_per_episode = np.zeros(episodes)
    
    previous_states = []
    max_previous_states = 5
    penalty = -0.2  # Penalización por movimiento repetitivo
    for i in range(episodes):
        state = env.reset()[0]  # Estado inicial
        terminated = False  # True cuando cae en un agujero o alcanza el objetivo
        truncated = False  # True cuando las acciones > 200
        previous_states_actions = []  # Inicializar lista de estados y acciones previas para el episodio

        while not terminated and not truncated:
            if is_training:
                if use_ucb:
                    # Selección de acción usando UCB
                    max_ucb = -np.inf
                    total_actions = np.sum(acciones[state, :])
                    for j in range(env.action_space.n):
                        ucb_value = q[state, j] + c * math.sqrt(math.log(total_actions + 1) / (acciones[state, j]))
                        if ucb_value > max_ucb:
                            max_ucb = ucb_value
                            action = j
                # En el caso de que no se use UCB
                else:
                    if rng.random() < epsilon:
                        action = env.action_space.sample()  # Acción aleatoria
                    else:
                        action = np.argmax(q[state, :])  # Seleccionar la acción con mayor valor estimado Q
            else:
                action = np.argmax(q[state, :])  # Seleccionar la acción con mayor valor estimado Q

            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if is_training:
                # Penalizar los movimientos hacia paredes (sin cambio de estado)
                if new_state == state:
                    reward -= 0.1
                    
                # Penalizar movimientos repetitivos entre dos estados
                if len(previous_states) >= max_previous_states:
                    previous_states.pop(0)
                previous_states.append((state, action))

                if previous_states.count((state, action)) > 1:
                    reward += penalty
                    
                # Actualización Incremental
                acciones[state, action] += 1
                recompensa = q[state, action]
                q[state, action] += alpha * (reward - recompensa)

            # Pasar la tabla Q y el recuento de episodios al entorno para renderizar
            if env.render_mode == 'human':
                env.set_q(q)
                env.set_episode(i)

            state = new_state
            rewards_per_episode[i] += reward
            
        #Disminución gradual de épsilon
        if is_training and not use_ucb:
            epsilon = max(epsilon - epsilon_decay, epsilon_min)
            
    env.close()
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa Acumulada por Episodio en Frozen Lake')
    plt.savefig('Laboratorios/Laboratorio7/Métodos/Gráficos/UCB/frozen_lake4x4_test.png')

    if is_training:
        with open("Laboratorios/Laboratorio7/Métodos/QTable/UCB/frozen_lake4x4.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(1000, is_training=False, render=True)
