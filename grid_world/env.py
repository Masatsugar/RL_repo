import enum
import random

import gym
import matplotlib.pyplot as plt
import numpy as np


class Action(enum.IntEnum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3


class CustomMaze(gym.Env):
    def __init__(self):

        self.states = [[i + 1, j + 1] for j in range(3) for i in range(4)]
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(12)

        self.walls = [
            [[1, 1], [1, 2]],
            [[1, 2], [2, 2]],
            [[2, 2], [2, 1]],
            [[2, 1], [1, 1]],
        ]
        self.init_policy = np.array(
            [
                [1, 1, np.nan, np.nan],  # S0
                [np.nan, 1, np.nan, 1],  # S1
                [1, 1, np.nan, 1],  # S2
                [1, np.nan, np.nan, 1],  # S3
                [1, np.nan, 1, np.nan],  # S4
                [np.nan, np.nan, np.nan, np.nan],  # S5
                [1, 1, 1, np.nan],  # S6
                [np.nan, np.nan, np.nan, np.nan],  # S7
                [np.nan, 1, 1, np.nan],  # S8
                [np.nan, 1, np.nan, 1],  # S9
                [np.nan, 1, 1, 1],  # S10
                [np.nan, np.nan, np.nan, np.nan],  # S11: Goal
            ]
        )
        self.start = 0
        self.goal = 11
        self.fire = 7
        self.cur_state = self.start
        self.done = False
        self.max_horizon = 200

    def reset(self):
        self.cur_state = self.start
        self.done = False
        self.counter = 0
        return self.cur_state

    def render(self, mode="human"):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        tick_params = {
            "axis": "both",
            "which": "both",
            "bottom": "off",
            "top": "off",
            "right": "off",
            "left": "off",
            "labelleft": "off",
            "labelbottom": "off",
        }

        for i in range(4):
            plt.plot([i, i], [0, 3], color="black", linewidth=1)
            plt.plot([0, 4], [i, i], color="black", linewidth=1)

        for wall in self.walls:
            l, r = wall
            plt.plot(l, r, color="red", linewidth=2)

        for i, state in enumerate(self.states):
            l, r = state
            plt.text(l - 0.5, r - 0.5, f"S{i}", size=14, ha="center")

        plt.text(0.5, 0.3, "START", ha="center")
        plt.text(3.5, 1.3, "Fire", ha="center")
        plt.text(3.5, 2.3, "GOAL", ha="center")
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 3)
        plt.tick_params(**tick_params)
        (line,) = ax.plot([0.5], [0.5], marker="o", color="g", markersize=60)
        plt.show()

    def step(self, action):
        if self.counter == self.max_horizon:
            return self.cur_state, 0.0, True, {}

        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this environment has already returned done = True. "
                "You should always call 'reset()' once you receive 'done = True' "
                "-- any further steps are undefined behavior."
            )
        if action == Action.Up:
            if self.cur_state in [0, 2, 3, 4, 6]:
                self.cur_state += 4
            if self.cur_state == self.fire:
                return self.cur_state, -1.0, True, {}
            return self.cur_state, 0.0, False, {}

        elif action == Action.Right:
            if self.cur_state in [0, 1, 2, 6, 8, 9, 10]:
                self.cur_state += 1
            if self.cur_state == self.goal:
                return self.cur_state, 1.0, True, {}
            if self.cur_state == self.fire:
                return self.cur_state, -1.0, True, {}
            return self.cur_state, 0.0, False, {}

        elif action == Action.Down:
            if self.cur_state in [4, 6, 8, 10]:
                self.cur_state -= 4
            return self.cur_state, 0.0, False, {}

        elif action == Action.Left:
            if self.cur_state in [1, 2, 3, 9, 10]:
                self.cur_state -= 1
            return self.cur_state, 0.0, False, {}
        else:
            raise ValueError("Action is ranged from [0, 1, 2, 3].")

    def compute_prob(self, state):
        pi = state.T / np.nansum(state, axis=1)
        return np.nan_to_num(pi.T)

    def goal_maze_ret_s_a(self, pi):
        state = 0
        s_a_history = [[state, np.nan]]
        total_reward = 0.0
        while 1:
            action = np.random.choice(
                [i for i in range(self.action_space.n)], p=pi[state, :]
            )
            state, reward, is_done, _ = self.step(action)
            total_reward += reward
            s_a_history.append([state, action])
            if is_done:
                print(f"Done: transition: {s_a_history}, total reward {total_reward}")
                break

        return s_a_history


class MarsRover(gym.Env):
    """
    Starnford CS234: Reinforcement Learning | Winter 2019 | Lecture 3 - Model-Free Policy Evaluation
        https://www.youtube.com/watch?v=dRIhrn8cc9w
    """

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(7)
        self.action_space = gym.spaces.Discrete(2)
        self.start = 3
        self.goal1 = 0
        self.goal2 = 6
        self.state = self.start
        self.slip_ratio = 0.3

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if random.random() > self.slip_ratio:
            if action == 0:
                self.state -= 1

            if action == 1:
                self.state += 1

        if self.state == self.goal1:
            return self.state, 1.0, True, {}
        elif self.state == self.goal2:
            return self.state, 10.0, True, {}
        else:
            return self.state, 0.0, False, {}


class AliasedGridWorld(gym.Env):
    """
    Grid world
    ----------
        | S0 | S1 | S2 | S3 | S4 |
        | S5 | S6 | S7 | S8 | S9 |

    """

    def __init__(self, start=0):
        self.start = start
        self.state = self.start
        self.observation_space = gym.spaces.Discrete(8)
        self.action_space = gym.spaces.Discrete(4)
        self.goal = 7
        self.fire = [5, 9]
        self.wall = [6, 8]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if action == Action.Up:
            if self.state in self.fire:
                self.state -= 5
            return self.state, 0.0, False, {}

        elif action == Action.Down:
            if self.state in [0, 2, 4]:
                self.state += 5
            if self.state == self.goal:
                return self.state, 1.0, True, {}

            if self.state in self.fire:
                return self.state, -1.0, True, {}

            return self.state, 0.0, False, {}

        elif action == Action.Right:
            if self.state in [0, 1, 2, 3]:
                self.state += 1
            return self.state, 0.0, False, {}
        else:
            if self.state in [1, 2, 3, 4]:
                self.state -= 1
            return self.state, 0.0, False, {}


if __name__ == "__main__":
    env = CustomMaze()
    env.reset()
    init_pi = env.compute_prob(env.init_policy)
    state_action = env.goal_maze_ret_s_a(init_pi)
