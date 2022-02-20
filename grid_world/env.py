import enum
from collections import Counter, defaultdict

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

    def compute_prob(self, state):
        pi = state.T / np.nansum(state, axis=1)
        return np.nan_to_num(pi.T)

    def get_next_state(self, pi, s):
        """次の状態を計算する

        Args:
            pi:
            s:

        Returns:
            next_state

        """

        next_direction = np.random.choice(
            [i for i in range(self.action_space.n)], p=pi[s, :]
        )
        if next_direction == Action.Up:
            a = 0
            s_next = s + 4
        elif next_direction == Action.Right:
            a, s_next = 1, s + 1
        elif next_direction == Action.Down:
            a, s_next = 2, s - 4
        elif next_direction == Action.Left:
            a, s_next = 3, s - 1
        else:
            raise ValueError("")
        return a, s_next

    def step(self, action):
        if self.counter == self.max_horizon:
            return self.cur_state, 0, True, {}

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
                return self.cur_state, -1, True, {}
            return self.cur_state, 0, False, {}

        elif action == Action.Right:
            if self.cur_state in [0, 1, 2, 6, 8, 9, 10]:
                self.cur_state += 1
            if self.cur_state == self.goal:
                return self.cur_state, 1, True, {}
            if self.cur_state == self.fire:
                return self.cur_state, -1, True, {}
            return self.cur_state, 0, False, {}

        elif action == Action.Down:
            if self.cur_state in [4, 6, 8, 10]:
                self.cur_state -= 4
            return self.cur_state, 0, False, {}

        elif action == Action.Left:
            if self.cur_state in [1, 2, 3, 9, 10]:
                self.cur_state -= 1
            return self.cur_state, 0, False, {}
        else:
            raise ValueError("Action is ranged from [0, 1, 2, 3].")

    def goal_maze_ret_s_a(self, pi):
        s = 0
        s_a_history = [[0, np.nan]]
        while 1:
            a, next_s = self.get_next_state(pi, s)
            s_a_history[-1][1] = a
            s_a_history.append([next_s.np.nan])
            if next_s == 8:
                print("Done")
                break
            else:
                s = next_s
        return s_a_history


if __name__ == "__main__":
    env = CustomMaze()
    env.reset()
