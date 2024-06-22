import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Register the enhanced frozen lake environment
gym.register(
    id="FrozenLake-enhanced",
    entry_point="frozen_lake_enhanced:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,
)

def run(episodes, is_training=True, render=False):
    # 'FrozenLake-enhanced' is the id specified above
    env = gym.make('FrozenLake-enhanced', desc=None, map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Inicializar la tabla Q
    else:
        with open('Laboratorios/Laboratorio7/Métodos/QTable/Incremental/frozen_lake8x8.pkl', 'rb') as f:
            q = pickle.load(f)
        
    epsilon = 1 # Probabilidad inicial de tomar acciones aleatorias
    epsilon_min = 0  # Valor mínimo de epsilon
    epsilon_decay = 0.0001 # 0.0001 * 1000 = 1 Luego de 10000 episodios el agente se convierte en greedy
    alpha = 0.1  # Tasa de aprendizaje
    rng = np.random.default_rng()  # Generador de números aleatorios

    rewards_per_episode = np.zeros(episodes)
    
    previous_states = []
    max_previous_states = 5
    penalty = -0.2  # Penalización por movimiento repetitivo
    
    for i in range(episodes):
        state = env.reset()[0]  # Estado inicial
        terminated = False  # True cuando cae en un agujero o alcanza el objetivo
        truncated = False  # True cuando las acciones > 200

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Acción aleatoria
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
                    
                # Actualización de Q-learning Incremental
                q[state, action] += alpha * (reward - q[state, action])

            # Pasar la tabla Q y el recuento de episodios al entorno para renderizar
            if env.render_mode == 'human':
                env.set_q(q)
                env.set_episode(i)

            state = new_state
            rewards_per_episode[i] += reward

        # Disminuir epsilon
        if is_training:
            epsilon = max(epsilon - epsilon_decay, epsilon_min)
            
    env.close()
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa Acumulada por Episodio en Frozen Lake')
    plt.savefig('Laboratorios/Laboratorio7/Métodos/Gráficos/Incremental/frozen_lake8x8_test.png')

    if is_training:
        with open("Laboratorios/Laboratorio7/Métodos/QTable/Incremental/frozen_lake8x8.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(1000, is_training=True, render=True)
    #run(1000, is_training=False, render=True)