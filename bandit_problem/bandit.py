import random
from typing import List, Any

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
    def __init__(self, arms, max_step=200):
        self.arms = arms
        self.max_step = max_step
        self.env_step = 0

    def step(self, action):
        self.env_step += 1
        done = True if self.env_step > self.max_step else False
        _reward = self.reward(action)
        return None, _reward, done, {}

    def reward(self, action):
        return self.arms[action].draw()


def test_algorithm(
    algo: Any, arms: List[BernoulliArm], num_sims: int = 200, horizon: int = 200,
) -> List[ndarray]:
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

    Returns:
        List[sim_nums, times, chosen_arms, rewards, cumulative_rewards]
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

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def run(algo: Any, label: str) -> None:
    """Run and plot results

    Parameters
    ----------
    algo
        Any algorithm for MAB
    label
        Key, the name of algorithm

    Returns
    -------
    """
    print(label)
    n_arms = len(theta)
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sims=NUM_SIMS, horizon=HORIZON)
    df = pd.DataFrame({"times": results[1], "rewards": results[3]})
    grouped = df["rewards"].groupby(df["times"])
    plt.plot(grouped.mean(), label=label)
    plt.legend(loc="best")


if __name__ == "__main__":
    NUM_SIMS = 200
    HORIZON = 400

    # 問題設定: 腕：7本のうち、あたりは1つ (0.8)とする。
    theta = [0.1, 0.4, 0.1, 0.2, 0.8, 0.1, 0.1]
    random.shuffle(theta)
    print(theta)
    arms = list(map(lambda x: BernoulliArm(x), theta))

    # Set Algorithms
    epsilons = [0.1]
    algos = {f"Eps={eps}": EpsilonGreedy(epsilon=eps) for eps in epsilons}
    algos.update(
        {
            "UCB": UCB1(),
            "TS": ThompsonSampling(),
            # "PG": PolicyGradient([], [], n_arms)
        }
    )
    for key, algo in algos.items():
        run(algo, label=key)

    plt.show()
