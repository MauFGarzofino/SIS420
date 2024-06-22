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
    env = gym.make('FrozenLake-enhanced', desc=None, map_name="4x4", is_slippery=False, render_mode='human' if render else None)
    
    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # Inicializamos la tabla Q
        acciones = np.ones((env.observation_space.n, env.action_space.n))  # Contador de acciones tomadas inicializado en 1 para evitar división por cero
    else:
        with open('Laboratorios/Laboratorio7/Métodos/QTable/UCB/frozen_lake4x4.pkl', 'rb') as f:
            q = pickle.load(f)
            acciones = np.ones((env.observation_space.n, env.action_space.n))
        
    c = 1.0  # Constante inicial para UCB
    c_min = 0  # Valor mínimo de c
    c_decay = 0.001
    alpha = 0.1  # Tasa de aprendizaje

    rewards_per_episode = np.zeros(episodes)
    
    previous_states = []
    max_previous_states = 4
    penalty = -0.2  # Penalización por movimiento repetitivo
    
    for i in range(episodes):
        state = env.reset()[0]  # Estado inicial
        terminated = False  # True cuando cae en un agujero o alcanza el objetivo
        truncated = False  # True cuando las acciones > 200

        while not terminated and not truncated:
            if is_training:
                # Selección de acción usando UCB con c variable
                total_actions = np.sum(acciones[state, :])
                ucb_values = q[state, :] + c * np.sqrt(np.log(total_actions + 1) / acciones[state, :])
                action = np.argmax(ucb_values)
            else:
                # Seleccionar la acción con mayor valor estimado Q
                action = np.argmax(q[state, :])

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
                acciones[state, action] += 1

            # Pasar la tabla Q y el recuento de episodios al entorno para renderizar
            if env.render_mode == 'human':
                env.set_q(q)
                env.set_episode(i)

            state = new_state
            rewards_per_episode[i] += reward

        # Disminuir c
        if is_training:
            c = max(c - c_decay, c_min)
    
    env.close()
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa Acumulada por Episodio en Frozen Lake')
    plt.savefig('Laboratorios/Laboratorio7/Métodos/Gráficos/UCB/frozen_lake4x4.png')
    plt.show()

    if is_training:
        with open("Laboratorios/Laboratorio7/Métodos/QTable/UCB/frozen_lake4x4.pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    #run(15000, is_training=True, render=True)
    run(1000, is_training=False, render=True)
