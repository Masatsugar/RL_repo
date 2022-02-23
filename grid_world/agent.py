import random
from collections import Counter, defaultdict
from typing import Optional, Tuple

import gym
import numpy as np
import scipy

from grid_world.env import MarsRover


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


class MonteCarlo:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.episode = 0

        self.counts = defaultdict(float)
        self.gs = defaultdict(float)
        self.values = defaultdict(float)
        self.history = None
        self.episodes = []

    def reset(self):
        self.history = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "done": [],
        }

    def play_episode(self):
        self.reset()
        env.reset()
        while True:
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            self.set_state(obs, action, reward, done)
            if done:
                self.episode += 1
                self.episodes.append(self.history.copy())
                self.reset()
                break

    def set_state(self, state, action, reward, done):
        self.history["obs"].append(state)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        self.history["done"].append(done)

    def discount_cumsum(self, rewards):
        return scipy.signal.lfilter(
            [1], [1, float(-self.gamma)], rewards[::-1], axis=0
        )[::-1]

    def policy_evaluation(self):
        for episode in self.episodes:
            rewards = self.discount_cumsum(episode["rewards"])
            for state, reward in zip(episode["obs"], rewards):
                self.counts[state] += 1
                self.gs[state] += reward
                self.values[state] += self.gs[state] / self.counts[state]

    def select_best_action(self, env):
        best_action, best_value = None, None
        for action in env.action_space.n:
            state, reward, done, _ = env.step(action)
            state_value = self.values[state]
            if best_value is None or best_value < state_value:
                best_value = state_value
                best_action = action
        return best_action


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env = MarsRover()
    mc = MonteCarlo(env)
    for i in range(10):
        mc.play_episode()

    mc.policy_evaluation()
