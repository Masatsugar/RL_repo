import random
from collections import Counter, defaultdict
from typing import Optional, Tuple

import gym
import numpy as np
import scipy.signal

from grid_world.env import MarsRover


class Exploration:
    def __init__(self, policy, epsilon=0.1):
        self.policy = policy
        self.epsilon = epsilon

    def greedy(self, state):
        best_action, best_value = None, None
        for action in range(self.policy.env.action_space.n):
            action_value = self.policy.action_values[state, action]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def epsilon_greedy(self, state):
        if random.random() > self.epsilon:
            return self.greedy(state)
        else:
            return self.policy.env.action_space.sample()


class Agent:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.state = self.env.reset()
        self.gamma = gamma
        self.epsilon = epsilon
        self.transitions = defaultdict(Counter)  # (state, action)
        self.reward_table = defaultdict(float)  # (state, action, next_state)
        self.value_table = defaultdict(float)

    def run_episode(self, env=None, is_random=False):
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

    def run_n_episodes(self, n=100, is_random=True):
        for _ in range(n):
            self.run_episode(is_random=is_random)

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

    def select_best_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.compute_state_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def compute_action(self, state):
        return self.select_best_action(state)

    def compute_state_value(self, state, action):
        """Update state values

            V(s) = sum ( P(s'|s, a) * (R(s, a) + gamma V(s')))

        In most of the cases, transition probability P is not known,
        so it is estimated from the path count of total count ratio.

        Examples:
            Assume that the observations are (s0, a0, s1), (s0, a0, s2).

            V(s0) = N(s1) / (N(s1) + N(s2)) (R(s0, a0, s1) + gamma V(s1))
                +  N(s2) / (N(s1) + N(s2)) (R(s0, a0, s2) + gamma V(s2))

        Parameters
        ----------
        state
        action

        Returns
        -------

        """
        counts = self.transitions[state, action]
        total = sum(counts.values())
        state_value = 0.0
        for target_state, count in counts.items():
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

    def select_best_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.value_table[state, action]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def compute_action(self, state):
        return self.select_best_action(state)

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                counts = self.transitions[state, action]
                total = sum(counts.values())
                for target_state, count in counts.items():
                    reward = self.reward_table[state, action, target_state]
                    best_action = self.compute_action(target_state)
                    action_value += (count / total) * (
                        reward
                        + self.gamma * self.value_table[target_state, best_action]
                    )
                self.value_table[state, action] = action_value


class TDlearning:
    def __init__(self, env, gamma=0.99, alpha=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.state = self.env.reset()
        self.state_values = defaultdict(float)
        self.action_values = defaultdict(float)
        self.episode = 0

    def sample(self) -> Tuple[np.ndarray, int, float, np.ndarray]:
        action = self.env.action_space.sample()
        cur_state = self.state
        new_state, reward, done, _ = self.env.step(action)
        self.state = self.env.reset() if done else new_state
        return cur_state, action, reward, new_state

    def update(self, state, action, reward, next_state):
        self.update_q(state, action, reward, next_state)
        self.update_v(state, action, reward, next_state)

    def update_q(self, state, action, reward, next_state):
        max_q, _ = self.best_action_value(next_state)
        td_target = reward + self.gamma * max_q
        cur_q = self.action_values[state, action]
        self.action_values[state, action] = cur_q + self.alpha * (
            td_target - cur_q
        )

    def update_v(self, state, action, reward, next_state):
        cur_v = self.state_values[state]
        next_v = self.state_values[next_state]
        td_target = reward + self.gamma * next_v
        self.state_values[state] = cur_v + self.alpha * (td_target - cur_v)

    def best_action_value(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.action_values[state, action]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def compute_action(self, state, episode: Optional[int] = None):
        if episode is not None:
            epsilon = 0.5 * (1 / (episode + 1))
            self.episode = episode
        else:
            epsilon = self.epsilon

        if random.random() > epsilon:
            _, action = self.best_action_value(state)
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
            self.update_q(state, action, reward, next_state)
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
        self.state_values = defaultdict(float)
        self.state_action_values = defaultdict(float)
        self.history = None
        self.episodes = []

    def reset(self):
        self.history = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "done": [],
        }

    def run_episode(self):
        self.reset()
        obs = self.env.reset()
        while True:
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            self.set_state(obs, action, reward, done)
            if done:
                self.episode += 1
                self.episodes.append(self.history.copy())
                self.reset()
                break
            obs = next_obs

    def set_state(self, state, action, reward, done):
        self.history["obs"].append(state)
        self.history["actions"].append(action)
        self.history["rewards"].append(reward)
        self.history["done"].append(done)

    def discount_cumsum(self, rewards):
        return scipy.signal.lfilter(
            [1], [1, float(-self.gamma)], rewards[::-1], axis=0
        )[::-1]

    def update_v(self, state, reward):
        self.counts[state] += 1
        self.gs[state] += reward
        self.state_values[state] = self.gs[state] / self.counts[state]

    def update_q(self, state, action, reward):
        self.counts[state, action] += 1
        self.gs[state, action] += reward
        self.state_action_values[state, action] = (
            self.gs[state, action] / self.counts[state, action]
        )

    def policy_evaluation(self):
        for episode in self.episodes:
            rewards = self.discount_cumsum(episode["rewards"])
            for state, action, reward in zip(
                episode["obs"], episode["actions"], rewards
            ):
                self.update_v(state, reward)
                self.update_q(state, action, reward)


if __name__ == "__main__":
    # env = gym.make("FrozenLake-v1")
    env = MarsRover()
    mc = MonteCarlo(env, gamma=1.0)

    for i in range(10):
        mc.run_episode()

    mc.policy_evaluation()

    # TD Learning
    env = MarsRover()
    td = TDlearning(env)
    for i in range(100):
        td.update(*td.sample())

    print(td.state_values)
    print(td.action_values)
