import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def softmax(x):
    x = x - np.max(x)  # Restar el valor máximo de x para evitar valores grandes en exp
    return np.exp(x) / np.sum(np.exp(x))

def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human' if render else None)
    
    alpha = 0.9  # Incrementar la tasa de aprendizaje
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    
    if is_training:
        H = np.zeros((env.observation_space.n, env.action_space.n))  # Inicializar la tabla H
    else:
        with open('frozen_lake4x4_H.pkl', 'rb') as f:
            H = pickle.load(f)
    
    for i in range(episodes):
        state = env.reset()[0]  # Estado inicial
        terminated = False
        truncated = False

        recompensas = []  # Lista para almacenar recompensas
        
        while not terminated and not truncated:
            pi = softmax(H[state])  # Calcular probabilidades iniciales para el estado actual
            action = rng.choice(range(env.action_space.n), p=pi)  # Seleccionar acción basada en probabilidades
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if is_training:
                # Penalizar los movimientos hacia paredes (sin cambio de estado)
                if new_state == state:
                    reward -= 0.5

                # Ajustar la recompensa para fomentar el comportamiento deseado
                if terminated and reward == 0:
                    reward = -1  # Penalizar caer en un agujero
                elif terminated and reward == 1:
                    reward = 10  # Recompensar alcanzar la meta
                else:
                    reward = 0.1  # Recompensa pequeña a movimientos seguros

                recompensas.append(reward)
                recompensa_media = np.mean(recompensas)  # Calcular recompensa media

                # Actualizar preferencias
                for j in range(env.action_space.n):
                    if j == action:
                        H[state, j] += alpha * (reward - recompensa_media) * (1 - pi[j])
                    else:
                        H[state, j] -= alpha * (reward - recompensa_media) * pi[j]

                # Clipping para evitar valores extremos en H
                H[state] = np.clip(H[state], -1000, 1000)

                # Imprimir tabla H actualizada
                if i % 1000 == 0:  # Imprimir cada 1000 episodios
                    print(f"Episodio: {i}, Estado: {state}, Acción: {action}, Recompensa: {reward}")
                    print(f"Tabla H actualizada:\n{H}\n")

            state = new_state
            rewards_per_episode[i] += reward

    env.close()
    
    # Visualización de la recompensa acumulada
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.plot(sum_rewards)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa Acumulada por Episodio en Frozen Lake')
    plt.savefig('frozen_lake4x4.png')
    plt.show()

    if is_training:
        with open("frozen_lake4x4_H.pkl", "wb") as f:
            pickle.dump(H, f)

        # Verificar que el archivo se haya guardado correctamente
        with open("frozen_lake4x4_H.pkl", "rb") as f:
            H_loaded = pickle.load(f)
            print("Tabla H cargada después de guardar:\n", H_loaded)

    # Evaluación de la política aprendida
    if not is_training:
        for _ in range(10):  # Ejecutar 10 episodios de evaluación
            state = env.reset()[0]
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = np.argmax(softmax(H[state]))  # Seleccionar la mejor acción basada en la política aprendida
                state, reward, terminated, truncated, _ = env.step(action)
                env.render()

if __name__ == '__main__':
    #run(1000, is_training=True, render=False)  # Aumentar el número de episodios de entrenamiento
    run(3, is_training=False, render=True)