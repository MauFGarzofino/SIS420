import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode='human')

def policy_evaluation(nb_iterations, policy, discount_factor=0.9, convergence_tolerance=10**(-6)):
    V_s = np.zeros(env.observation_space.n)
    for iteration in range(nb_iterations):
        V_s_next = np.zeros(env.observation_space.n)
        for state in env.P:
            action_sum = 0
            for action in env.P[state]:
                state_sum = 0
                for proba_next, next_state, reward, is_final_state in env.P[state][action]:
                    state_sum += proba_next * (reward + discount_factor * V_s[next_state])
                action_sum += policy[state][action] * state_sum
            V_s_next[state] = action_sum
        if np.max(np.abs(V_s_next - V_s)) < convergence_tolerance:
            V_s = V_s_next
            print('policy evaluation iteration finished !!')
            break
        V_s = V_s_next
    return V_s


def policy_iteration(nb_iterations, discount_factor=0.9):
    # Initializing random policy
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    while True:
        # Compute policy evaluation
        V_states = policy_evaluation(nb_iterations=nb_iterations, policy=policy)

        policy_stable = True

        for state in range(env.observation_space.n):
            simple_action = np.argmax(policy[state])

            actions_values = np.zeros(env.action_space.n)
            for action in range(len(actions_values)):
                for proba_next, next_state, reward, is_final_state in env.P[state][action]:
                    actions_values[action] += proba_next * (reward + discount_factor * V_states[next_state])
            best_action = np.argmax(actions_values)

            if simple_action != best_action:
                policy_stable = False

            # the policy 2D array at the index 'state' is equal to an array of 0s expect a 1 at the index 'best action'
            policy[state] = np.eye(env.action_space.n)[best_action]

        if policy_stable:
            return policy, V_states


def ResolveFrozenLake():
    policy, V_states = policy_iteration(1000)
    env.reset()

    is_final_state = False
    current_state = 0
    while True:
        if is_final_state == True:
            if current_state != 15:
                print('WE LOST :(')
            else:
                print('WE WON!!')
            break

        observations = env.step(np.argmax(policy[current_state]))
        current_state = observations[0]
        is_final_state = observations[2]
        env.render()

    env.close()

if __name__ == '__main__':
    nb_iterations = 100
    # Observation space - states
    states = env.observation_space
    number_of_states = states.n

    # Action space - actions
    actions = env.action_space
    number_of_actions = actions.n

    # Define random policy
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    # Iterative Policy Evaluation
    V_states = policy_evaluation(nb_iterations=nb_iterations, policy=policy)
    #print(V_states)

    # Policy Iteration (policy evaluation + policy improvement)
    policy, V_states = policy_iteration(nb_iterations)
    #print(policy)

    ResolveFrozenLake()