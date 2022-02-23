import random
from collections import Counter, defaultdict
from typing import Optional, Tuple

import gym
import numpy as np


class Agent:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.state = self.env.reset()
        self.gamma = gamma
        self.epsilon = epsilon
        self.transitions = defaultdict(Counter)  # (state, action)
        self.reward_table = defaultdict(float)  # (state, action, next_state)
        self.value_table = defaultdict(float)

    def play_episode(self, env=None, is_random=False):
        total_reward = 0.0
        if env is None:
            env = self.env

        state = env.reset()
        while True:
            action = (
                env.action_space.sample() if is_random else self.compute_action(state)
            )
            next_state, reward, is_done, _ = env.step(action)
            self.set_state_action(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            if is_done:
                break

        return total_reward

    def play_random_steps(self, n=100, is_random=True):
        for _ in range(n):
            self.play_episode(is_random=is_random)

    def compute_action(self, state):
        raise NotImplementedError

    def set_state_action(self, state, action, reward, next_state):
        self.reward_table[state, action, next_state] = reward
        self.transitions[state, action][next_state] += 1


class Vlearning(Agent):
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        super(Vlearning, self).__init__(env, gamma, epsilon)
        self.env = env
        self.state = self.env.reset()

    def compute_action(self, state):
        if random.random() > self.epsilon:
            best_action, best_value = None, None
            for action in range(self.env.action_space.n):
                action_value = self.compute_state_value(state, action)
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            return best_action
        else:
            return self.env.action_space.sample()

    def compute_state_value(self, state, action):
        target_counts = self.transitions[state, action]
        total = sum(target_counts.values())
        state_value = 0.0
        for target_state, count in target_counts.items():
            reward = self.reward_table[state, action, target_state]
            state_value += (count / total) * (
                reward + self.gamma * self.value_table[target_state]
            )
        return state_value

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.compute_state_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.value_table[state] = max(state_values)


class Qlearning(Agent):
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        super(Qlearning, self).__init__(env, gamma, epsilon)
        self.env = env
        self.state = self.env.reset()

    def compute_action(self, state):
        if random.random() > self.epsilon:
            best_action, best_value = None, None
            for action in range(self.env.action_space.n):
                action_value = self.value_table[state, action]  # select max Q-value
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            return best_action
        else:
            return self.env.action_space.sample()

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transitions[state, action]
                total = sum(target_counts.values())
                for target_state, count in target_counts.items():
                    reward = self.reward_table[state, action, target_state]
                    best_action = self.compute_action(target_state)
                    action_value += (count / total) * (
                        reward
                        + self.gamma * self.value_table[target_state, best_action]
                    )
                self.value_table[state, action] = action_value


class Qlearning2:
    def __init__(self, env, gamma=0.99, alpha=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.state = self.env.reset()
        self.value_table = defaultdict(float)
        self.episode = 0

    def sample(self) -> Tuple[np.ndarray, int, float, np.ndarray]:
        action = self.env.action_space.sample()
        cur_state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = self.env.reset() if done else new_state
        return cur_state, action, reward, new_state

    def update(self, state, action, reward, next_state):
        max_q, _ = self.best_value_action(next_state)
        td_target = reward + self.gamma * max_q
        cur_q = self.value_table[state, action]
        self.value_table[state, action] = cur_q + self.alpha * (td_target - cur_q)

    def best_value_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.value_table[state, action]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def compute_action(self, state, episode: Optional[int] = None):
        if episode is not None:
            epsilon = 0.5 * (1 / (episode + 1))
            self.epsilon = episode
        else:
            epsilon = self.epsilon

        if random.random() > epsilon:
            _, action = self.best_value_action(state)
        else:
            action = self.env.action_space.sample()
        return action

    def run_episode(self, env=None, episode=None):
        if env is None:
            env = self.env
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.compute_action(state, episode=episode)
            next_state, reward, done, _ = env.step(action)
            self.update(state, action, reward, next_state)
            total_reward += reward
            if done:
                break
            state = next_state

        return total_reward
