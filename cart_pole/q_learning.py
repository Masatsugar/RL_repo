from collections import Counter
from queue import Queue
from typing import Tuple

import gym
import numpy as np


class QTable:
    def __init__(self, env, num_digit=6, init_qtable="random"):
        """
        Observation bounds:
            cart_x = (-2.4, 2.4)
            cart_v = (-Inf, Inf)
            pole_angle = (-41.8°, 41.8°)
            pole_v = (-Inf, Inf)
        """
        self.env = env
        self.num_digit = num_digit

        if init_qtable == "random":
            self.q_table = np.random.uniform(
                low=0,
                high=1,
                size=(num_digit ** env.observation_space.shape[0], env.action_space.n),
            )
        else:
            self.q_table = np.zeros(
                shape=(num_digit ** env.observation_space.shape[0], env.action_space.n)
            )

        self.bound = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.5, 0.5], [-2.0, 2.0]])
        self.shape = self.q_table.shape

    def digitize(self, observation: np.ndarray) -> int:
        """
        Returns discrete state from observation.

        This method digitizes the continuous state into discrete state.

        Parameters
        ----------
        observation

        Returns
        -------
            state index

        """
        bins = lambda x, y, z: np.linspace(x, y, z + 1)[1:-1]
        bins_list = [
            bins(x, y, self.num_digit)
            for x, y in zip(self.bound[:, 0], self.bound[:, 1])
        ]
        digit = [np.digitize(obs, lst) for obs, lst in zip(observation, bins_list)]
        return sum([dig * (self.num_digit ** i) for i, dig in enumerate(digit)])

    def __getitem__(self, idx):
        return self.q_table[idx]

    def __setitem__(self, key, value):
        self.q_table[key] = value

    def __repr__(self):
        return f"{self.__class__.__name__}(env={self.env}, num_digits={self.num_digit}, q_table={self.q_table})"


class Action:
    def __init__(self, env):
        self.num_action = env.action_space.n
        self.episode = 0

    def greedy(self, q_table: QTable, state: int) -> int:
        return int(np.argmax(q_table[state]))

    def epsilon_greedy(self, q_table: QTable, state: int, episode: int) -> int:
        self.episode = episode
        epsilon = 0.5 * (1 / (episode + 1))
        if np.random.uniform(0, 1) > epsilon:
            action = self.greedy(q_table, state)
        else:
            action = np.random.choice(self.num_action)
        return action


class QLearning:
    def __init__(self, env, num_digit, alpha=0.5, gamma=0.99, init_qtable="random"):
        self.action = Action(env)
        self.q_table = QTable(env, num_digit=num_digit, init_qtable=init_qtable)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, state, action, reward, next_state) -> None:
        """Off-policy update

        Temporal difference target:
            TD = reward_{t} + \gamma * max_a Q(s_{t+1}, a)

        Update rule:
            Q(s_t, a_t) := Q(s_t, a_t) + \alpha * ( TD - Q(s_t, a_t) )

        Parameters
        ----------
        state
        action
        reward
        next_state

        Returns
        -------

        """
        state = self.q_table.digitize(state)
        next_state = self.q_table.digitize(next_state)
        cur_q = self.q_table[state, action]
        max_q = max(self.q_table[next_state][:])
        td_target = reward + self.gamma * max_q
        self.q_table[state, action] = cur_q + self.alpha * (td_target - cur_q)

    def compute_action(self, observation, episode: int) -> int:
        state = self.q_table.digitize(observation)
        return self.action.epsilon_greedy(
            q_table=self.q_table, state=state, episode=episode
        )


class Sarsa:
    def __init__(self, env, num_digit, alpha=0.5, gamma=0.99):
        self.action = Action(env)
        self.q_table = QTable(env, num_digit=num_digit)
        self.alpha = alpha
        self.gamma = gamma

    def update(self, state, action, reward, next_state) -> None:
        """On-policy update

        A difference between Q-learning and SARSA is to use Q(s_t, a_t) instead of using max Q(s_t, a_t).
        Temporal difference:

            a_{t+1} = \pi (s_{t+1})
            TD = reward + \gamma * Q(s_{t+1}, a_{t+1})

        Actual next action $a_{t+1}$ is obtained from the current Q table.

        Update rule:
            Q(s_t, a_t) := Q(s_t, a_t) + eta * ( TD - Q(s_t, a_t))

        Parameters
        ----------
        state
        action
        reward
        next_state

        Returns
        -------

        """
        # select next action from policy.
        next_action = self.compute_action(next_state, self.action.episode)

        state = self.q_table.digitize(state)
        next_state = self.q_table.digitize(next_state)

        cur_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q
        self.q_table[state, action] = cur_q + self.alpha * (td_target - cur_q)

    def compute_action(self, observation: np.ndarray, episode: int) -> int:
        state = self.q_table.digitize(observation)
        return self.action.epsilon_greedy(
            q_table=self.q_table, state=state, episode=episode
        )


class DoubleQLearning:
    """
    Double Q-learning, Hado van Hasselt, NIPS2010
        https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf

    """

    def __init__(
        self, env, num_digit, alpha=1.0, gamma=0.99, degree=1.0, init_qtable="random"
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.degree = degree
        self.qa = QLearning(env, num_digit, alpha, gamma, init_qtable)
        self.qb = QLearning(env, num_digit, alpha, gamma, init_qtable)
        self.counter = Counter()

    def update(self, state, action, reward, next_state):
        state = self.qa.q_table.digitize(state)
        next_state = self.qa.q_table.digitize(next_state)
        self.counter[state, action] += 1
        alpha = self.alpha / np.power(self.counter[state, action], self.degree)
        if np.random.random() >= 0.5:
            # update Q-A
            cur_q = self.qa.q_table[state, action]
            action = np.argmax(self.qa.q_table[next_state])
            td_target = (
                reward + self.gamma * self.qb.q_table[next_state, action]
            )  # use QB!
            self.qa.q_table[state, action] = cur_q + alpha * (td_target - cur_q)
        else:
            # update Q-B
            cur_q = self.qb.q_table[state, action]
            action = np.argmax(self.qb.q_table[next_state])
            td_target = (
                reward + self.gamma * self.qa.q_table[next_state, action]
            )  # use QA!
            self.qb.q_table[state, action] = cur_q + alpha * (td_target - cur_q)

    def compute_action(self, observation, episode):
        state = self.qa.q_table.digitize(observation)
        if max(self.qa.q_table[state]) > max(self.qb.q_table[state]):
            q_table = self.qa.q_table[state]
        else:
            q_table = self.qb.q_table[state]
        # q_table = self.qa.q_table[state] + self.qb.q_table[state]
        num_actions = len(q_table)
        n_visited_states = sum(
            [self.counter[state, action] for action in range(num_actions)]
        )
        if n_visited_states == 0:
            epsilon = 1.0
        else:
            epsilon = 1.0 / np.sqrt(n_visited_states)
        if np.random.uniform(0, 1) >= epsilon:
            action = np.argmax(q_table)
        else:
            action = np.random.choice(num_actions)
        return action


class Agent:
    def __init__(self, env, episode=500, horizon=200):
        self.env = env
        self.episode = episode
        self.horizon = horizon

        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def reward(
        self, done: bool, step: int, complete_episodes: int
    ) -> Tuple[float, int]:
        """
        Custom reward function

        Parameters
        ----------
        done
        step
        complete_episodes

        Returns
        -------
        reward
        complete_episodes

        """
        if done:
            if step < 195:
                reward = -1.0
                complete_episodes = 0
            else:
                reward = 1.0
                complete_episodes += 1
        else:
            reward = 0.0

        return reward, complete_episodes

    @staticmethod
    def record(
        episode, episode_reward, global_ep_reward, result_queue, total_loss, num_steps
    ):
        if global_ep_reward == 0.0:
            global_ep_reward = episode_reward
        else:
            global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
        print(
            f"Episode: {episode} | "
            f"Moving Average Reward: {int(global_ep_reward)} | "
            f"Episode Reward: {int(episode_reward)} | "
            f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
            f"Steps: {num_steps} | "
        )
        result_queue.put(global_ep_reward)
        return global_ep_reward

    def run(self, algo):
        reward_mean = 0.0
        complete_episodes = 0
        for episode in range(self.episode):
            obs = self.env.reset()
            total_reward = 0.0
            for step in range(self.horizon):
                action = algo.compute_action(obs, episode=episode)
                next_obs, reward, done, _ = self.env.step(action)

                # takes an immediate custom reward
                my_reward, complete_episodes = self.reward(
                    done, step, complete_episodes
                )
                algo.update(obs, action, my_reward, next_obs)
                obs = next_obs
                if done:
                    print(f"{episode} Episode : Finished after {step+1} time steps")
                    break

                self.global_moving_average_reward = self.record(
                    episode=episode,
                    episode_reward=total_reward,
                    global_ep_reward=self.global_moving_average_reward,
                    result_queue=self.res_queue,
                    total_loss=0,
                    num_steps=step + 1,
                )
                total_reward += reward

                reward_mean += total_reward

            final_avg = reward_mean / float(self.episode)
            print(f"Average score across {self.episode} episodes: {final_avg}")

            if complete_episodes >= 5:
                print("5 times successes")

    def play_episodes(self, algo):
        complete_episodes = 0
        for episode in range(self.episode):
            state = env.reset()
            total_reward = 0.0
            for step in range(self.horizon):
                action = algo.compute_action(state, episode=episode)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                my_reward, complete_episodes = self.reward(
                    done, step, complete_episodes=complete_episodes
                )
                algo.update(state, action, my_reward, next_state)
                if done:
                    if episode % 10 == 0:
                        print(f"episode={episode}, total_reward={total_reward}")
                    break
                state = next_state

            if complete_episodes >= 10:
                print("10 times successes")
                break


if __name__ == "__main__":
    from tqdm import tqdm

    env = gym.make("CartPole-v0")
    q_learning = QLearning(env, num_digit=9, alpha=0.5, gamma=0.99)
    sarsa = Sarsa(env, num_digit=5, alpha=0.5, gamma=0.99)
    agent = Agent(env, episode=1000)
    double_q_learning = DoubleQLearning(env, num_digit=6, alpha=1.0, gamma=0.95)
    agent.play_episodes(double_q_learning)
