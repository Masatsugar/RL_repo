import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import EpsilonGreedy, UCB1, ThompsonSampling
from utils import BernoulliArm


def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    rewards = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    sim_nums = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)

    for sim in tqdm(range(num_sims)):
        sim = sim + 1
        algo.initialize(len(arms))

        for t in range(horizon):
            t = t + 1
            index = (sim - 1) * horizon + t - 1
            sim_nums[index] = sim
            times[index] = t
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm

            reward = arms[chosen_arm].draw()
            rewards[index] = reward

            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)

    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def run(algo, label):
    print(label)
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sims=NUM_SIMS, horizon=HORIZON)

    df = pd.DataFrame({"times": results[1], "rewards": results[3]})
    grouped = df["rewards"].groupby(df["times"])
    plt.plot(grouped.mean(), label=label)
    plt.legend(loc="best")


if __name__ == '__main__':
    NUM_SIMS = 100
    # 選択数設定
    HORIZON = 1000
    # 試行回数設定

    # 問題設定: 腕：50本のうち、あたりは1つ (0.8)とする。
    theta = np.array([0.07383118, 0.1269802, 0.05189434, 0.01803404, 0.12071388, 0.06441495, 0.00693751, 0.05370886, 0.00197787, 0.02400634, 0.08780609, 0.04508783, 0.07516995, 0.10146196, 0.06962517, 0.01745689, 0.26292255, 0.14591762, 0.12352253, 0.03706591, 0.12318411, 0.05474484, 0.16220637, 0.07945365, 0.03227532, 0.02607283, 0.14865122, 0.03190347, 0.00351194, 0.0401763, 0.46733392, 0.04837297, 0.02435725,
                      0.03607217, 0.01343893, 0.12936847, 0.05900506, 0.10996883, 0.18994591, 0.05006086, 0.03869274, 0.03478122, 0.05367832, 0.01793802, 0.00050172, 0.06897512, 0.00293645, 0.07149655, 0.49545609, 0.04973641, 0.03508565, 0.03436505, 0.05315849, 0.04215852, 0.1810515, 0.16025654, 0.0346163, 0.07989476, 0.01696486, 0.01413587, 0.06926118, 0.02459153, 0.21603212, 0.20380086, 0.03113274, 0.03491508])
    n_arms = len(theta)
    # random.shuffle(theta)
    print(theta)

    arms = map(lambda x: BernoulliArm(x), theta)
    arms = list(arms)

  # EpsilonGreedyのepsilonの値設定
    algos = {'Eps': EpsilonGreedy([], [], epsilon=0.05), 'UCB': UCB1(
        [], []), 'TS': ThompsonSampling([], [])}
    for key, algo in algos.items():
        run(algo, label=key)

    plt.show()
