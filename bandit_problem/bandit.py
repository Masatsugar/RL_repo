import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm

from bandit_problem.utils import (UCB1, BernoulliArm, EpsilonGreedy,
                                  PolicyGradient, ThompsonSampling)


def test_algorithm(
    policy: Any, arms: List[BernoulliArm], num_sims: int = 200, horizon: int = 200,
) -> Dict[str, ndarray]:
    """Run an algorithm for evaluation in MAB

    Parameters
    ----------
    policy
        A policy of action for MAB: EpsilonGreedy, UCB1, ThompsonSampling, or Softmax.
    arms
        Bernoulli arm's instance list.
    num_sims
        The number of simulations (episodes in RL settings)
    horizon
        The number of drawing arms in MAB.

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
        policy.reset(len(arms))
        for step in range(horizon):
            step += 1
            index = (sim - 1) * horizon + step - 1
            sim_nums[index] = sim
            times[index] = step

            # select an arm and obtain the reward.
            chosen_arm = policy.select_arm()
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
    algo.reset(n_arms)
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

    # 問題設定: 腕：K本のうち、あたりは1つ (0.9)とする。
    theta = [0.1, 0.1, 0.4, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    # random.shuffle(theta)
    print(theta)
    arms = list(map(lambda x: BernoulliArm(x), theta))

    # Set Algorithms
    policy_list = [
        EpsilonGreedy(epsilon=0.1),
        EpsilonGreedy(epsilon=0.3),
        UCB1(),
        # ThompsonSampling(),
        # PolicyGradient(len(arms))
    ]
    plt.figure(dpi=300)
    for policy in policy_list:
        run(arms, policy)
    # plt.savefig("mab.png")
    plt.show()
    plt.clf()
