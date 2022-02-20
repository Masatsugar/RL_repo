from collections import defaultdict, Counter
import random


class Vlearning:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.state = self.env.reset()
        self.transitions = defaultdict(Counter)  # (state, action)
        self.reward_table = defaultdict(float)  # (state, action, next_state)
        self.value_table = defaultdict(float)
        self.epsilon = epsilon

    def play_episode(self, env):
        total_reward = 0.0
        state = self.env.reset()
        while 1:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.reward_table[state, action, new_state] = reward
            self.transitions[state, action][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def select_action(self, state):
        if random.random() > self.epsilon:
            best_action, best_value = None, None
            for action in range(self.env.action_space.n):
                action_value = self.calculate_action_value(state, action)
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            return best_action
        else:
            return self.env.action_space.sample()

    def play_random_steps(self, n=100):
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.reward_table[self.state, action, new_state] = reward
            self.transitions[self.state, action][new_state] += 1
            if is_done:
                self.state = self.env.reset()
            else:
                self.state = new_state

    def calculate_action_value(self, state, action):
        target_counts = self.transitions[state, action]
        total = sum(target_counts.values())
        action_value = 0.0
        for target_state, count in target_counts.items():
            reward = self.reward_table[state, action, target_state]
            action_value += (count / total) * (
                reward + self.gamma * self.value_table[target_state]
            )
        return action_value

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.calculate_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.value_table[state] = max(state_values)


# import numpy as np
#
# states = np.arange(0, 12)
# for state in states:
#     counts = []
#     for action in [0, 1, 2, 3]:
#         target_counts = self.transitions[state, action]
#         total = sum(target_counts.values())
#         counts.append(total)
#
#     probs = np.array(counts) / sum(counts)
#     policy[state] = probs
