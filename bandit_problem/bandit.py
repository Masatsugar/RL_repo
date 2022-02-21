import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm

from bandit_problem.utils import (
    UCB1,
    BernoulliArm,
    EpsilonGreedy,
    PolicyGradient,
    ThompsonSampling,
)


class MultiArmedBandit:
    def __init__(self, arms: List[BernoulliArm], max_step: int = 200):
        self.arms = arms
        self.max_step = max_step
        self.env_step = 0
        self.cumulative_reward = 0
        self.trajectory = []

    def reset(self):
        self.env_step = 0
        self.cumulative_reward = 0

    def step(self, action):
        self.env_step += 1
        reward = self.reward(action)
        self.cumulative_reward += reward
        done = True if self.env_step > self.max_step else False
        return None, reward, done, {}

    def reward(self, action):
        return self.arms[action].draw()

    def run(self, algo):
        self.reset()
        algo.initialize(len(self.arms))
        for i in range(self.max_step):
            action = algo.select_arm()
            _, reward, done, _ = self.step(action)
            self.trajectory.append(action)
            algo.update(action, reward)
            if done:
                break
        print(f"done: reward mean={self.cumulative_reward / self.max_step}")
        return self


def test_algorithm(
    algo: Any, arms: List[BernoulliArm], num_sims: int = 200, horizon: int = 200,
) -> Dict[str, ndarray]:
    """Run an algorithm for evaluation in MAB

    Parameters
    ----------
    algo
        RL Algorithm for MAB: EpsilonGreedy, UCB1, ThompsonSampling, or Policy Gradient.
    arms
        Bernoulli Arms instance
    num_sims
        The number of simulations (episodes in RL settings)
    horizon
        The number of drawing MAB.

    Returns
        Dict: {sim_nums, times, chosen_arms, rewards, cumulative_rewards}
    -------
    """

    # Initialize variables with zeros.
    chosen_arms = np.zeros(num_sims * horizon)
    rewards = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    sim_nums = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)

    for sim in tqdm(range(num_sims)):
        sim += 1
        algo.initialize(len(arms))
        for step in range(horizon):
            step += 1
            index = (sim - 1) * horizon + step - 1
            sim_nums[index] = sim
            times[index] = step

            # select an arm and obtain the reward.
            chosen_arm = algo.select_arm()
            reward = arms[chosen_arm].draw()

            # store the choice.
            chosen_arms[index] = chosen_arm
            rewards[index] = reward
            cumulative_rewards[index] = (
                reward if step == 1 else cumulative_rewards[index - 1] + reward
            )

            # update algorithm
            algo.update(chosen_arm, reward)

    return {
        "sim_nums": sim_nums,
        "times": times,
        "chosen_arms": chosen_arms,
        "rewards": rewards,
        "cumulative_rewards": cumulative_rewards,
    }


def run(arms: List[BernoulliArm], algo: Any, y_label: str = "rewards") -> None:
    """Run and plot results

    Parameters
    ----------
    algo
        Any algorithm for MAB
    y_label
        "rewards", "chosen_arms", or "cumulative_rewards"

    Returns
    -------
    """
    label_name = algo.__class__.__name__
    if label_name == "EpsilonGreedy":
        label_name += f"({algo.epsilon})"
    print(label_name)
    n_arms = len(arms)
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sims=NUM_SIMS, horizon=HORIZON)
    df = pd.DataFrame(results)
    grouped = df[y_label].groupby(df["times"])

    # Figure
    plt.title(f"Multi-armed bandit: sims={NUM_SIMS}")
    plt.plot(grouped.mean(), label=label_name)
    plt.ylabel(f"{y_label} mean")
    plt.xlabel("Number of steps")
    plt.legend(loc="best")
    # plt.show()


if __name__ == "__main__":
    NUM_SIMS = 200
    HORIZON = 2000

    # 問題設定: 腕：7本のうち、あたりは1つ (0.8)とする。
    theta = [0.1, 0.1, 0.4, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    # random.shuffle(theta)
    print(theta)
    arms = list(map(lambda x: BernoulliArm(x), theta))

    # Set Algorithms
    algo_list = [
        EpsilonGreedy(epsilon=0.1),
        EpsilonGreedy(epsilon=0.3),
        UCB1(),
        # ThompsonSampling(),
        # PolicyGradient(len(arms))
    ]
    plt.figure(dpi=300)
    for algo in algo_list:
        run(arms, algo)
    # plt.savefig("mab.png")
    plt.show()
    plt.clf()

    # Another Example
    env = MultiArmedBandit(arms=arms, max_step=HORIZON)
    env.reset()
    algo = UCB1()  # EpsilonGreedy(epsilon=0.2)
    env.run(algo=algo)
    print(f"Counts of chosen arms={algo.counts}, \nreward_mean={algo.values}")

    plt.plot(range(0, HORIZON), env.trajectory, "x-")
    plt.title(algo.__class__.__name__)
    plt.xlabel("step")
    plt.ylabel("arm")
    plt.show()
