from queue import Queue
from typing import Tuple

import gym
import numpy as np


class QTable:
    def __init__(self, env, num_digit):
        """
        cart_x = (-2.4, 2.4)
        cart_v = (-Inf, Inf)
        pole_angle = (-41.8°, 41.8)
        pole_v = (-Inf, Inf)
        """
        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(num_digit ** env.observation_space.shape[0], env.action_space.n),
        )
        self.num_digit = num_digit
        self.bound = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.5, 0.5], [-2.0, 2.0]])

    def digitize(self, observation: np.ndarray) -> int:
        """
        Digitizes the continuous state into discrete state

        Parameters
        ----------
        observation

        Returns
        -------

        """
        bins = lambda x, y, z: np.linspace(x, y, z + 1)[1:-1]
        bins_list = list(
            map(
                lambda x, y: bins(x, y, self.num_digit),
                self.bound[:, 0],
                self.bound[:, 1],
            )
        )
        # bins_list = [bins(x, y, self.num.digit) for x, y in zip(bound[:, 0], bound[,:1])]
        digit = [np.digitize(obs, lst) for obs, lst in zip(observation, bins_list)]
        return sum([dig * (self.num_digit ** i) for i, dig in enumerate(digit)])

    def __getitem__(self, idx):
        return self.q_table[idx]

    def __setitem__(self, key, value):
        self.q_table[key] = value

    def __repr__(self):
        return self.q_table


class Action:
    def __init__(self, env):
        self.num_action = env.action_space.n
        self.episode = 0

    def greedy(self, q_table, state):
        return np.argmax(q_table[state][:])

    def epsilon_greedy(self, q_table, state, episode) -> int:
        self.episode = episode
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon < np.random.uniform(0, 1):
            action = self.greedy(q_table, state)
        else:
            action = np.random.choice(self.num_action)
        return action


class QLearning:
    def __init__(self, env, num_digit, eta=0.5, gamma=0.99):
        self.action = Action(env)
        self.q_table = QTable(env, num_digit=num_digit)
        self.eta = eta
        self.gamma = gamma

    def update(self, state, action, reward, next_state) -> None:
        """Off-policy update

        max_a Q(s_{t+1}, a_t)
        Q(s_t, a_t) := Q(s_t, a_t) + eta * ( reward + gamma * max Q(s_{t+1}, a_t) - Q(s_t, a_t))

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
        self.q_table[state, action] = cur_q + self.eta * (
            reward + self.gamma * max_q - cur_q
        )

    def compute_action(self, observation, episode: int) -> int:
        state = self.q_table.digitize(observation)
        return self.action.epsilon_greedy(
            q_table=self.q_table, state=state, episode=episode
        )


class Sarsa:
    def __init__(self, env, num_digit, eta=0.5, gamma=0.99):
        self.action = Action(env)
        self.q_table = QTable(env, num_digit=num_digit)
        self.eta = eta
        self.gamma = gamma

    def update(self, state, action, reward, next_state) -> None:
        """On-policy update

        A difference between Q-learning and SARSA is to use Q(s_t, a_t) instead of using max Q(s_t, a_t).
            Q(s_t, a_t) := Q(s_t, a_t) + eta * ( reward + gamma * Q(s_{t+1}, a_{t+1} - Q(s_t, a_t))

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

        # select next action from policy.
        next_action = self.action.epsilon_greedy(
            self.q_table, next_state, self.action.episode
        )
        cur_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        self.q_table[state, action] = cur_q + self.eta * (
            reward + self.gamma * next_q - cur_q
        )

    def compute_action(self, observation, episode: int) -> int:
        state = self.q_table.digitize(observation)
        return self.action.epsilon_greedy(
            q_table=self.q_table, state=state, episode=episode
        )


class Agent:
    def __init__(self, env, episode=500, horizon=200):
        self.env = env
        self.EPISODE = episode
        self.HORIZON = horizon

        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    def reward(self, done: bool, step: int, complete_episodes: int) -> Tuple[int, int]:
        if done:
            if step < 195:
                reward = -1
                complete_episodes = 0
            else:
                reward = 1
                complete_episodes += 1
        else:
            reward = 0

        return reward, complete_episodes

    @staticmethod
    def record(
        episode, episode_reward, global_ep_reward, result_queue, total_loss, num_steps
    ):
        if global_ep_reward == 0:
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
        reward_avg = 0.0
        complete_episodes = 0
        for episode in range(self.EPISODE):
            obs = self.env.reset()
            reward_sum = 0.0
            for step in range(self.HORIZON):
                action = algo.compute_action(obs, episode=episode)
                next_obs, reward, done, _ = self.env.step(action)

                # takes an immediate reward
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
                    episode_reward=reward_sum,
                    global_ep_reward=self.global_moving_average_reward,
                    result_queue=self.res_queue,
                    total_loss=0,
                    num_steps=step + 1,
                )

                reward_sum += reward
                reward_avg += reward_sum
                final_avg = reward_avg / float(self.EPISODE)
                print(f"Average score across {self.EPISODE} episodes: {final_avg}")

            if complete_episodes >= 5:
                print("5 times successes")


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    alg1 = QLearning(env, num_digit=6, eta=0.5, gamma=0.99)
    alg2 = Sarsa(env, num_digit=6, eta=0.5, gamma=0.99)
    agent = Agent(env, episode=200)
    agent.run(alg1)
