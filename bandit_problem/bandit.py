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
    theta = np.array([0.02234092, 0.0266909, 0.21540886, 0.00616458, 0.2541345, 0.03962746, 0.06908766, 0.19013299, 0.09385227, 0.03494758, 0.03838679, 0.07670622, 0.02583836, 0.33198215, 0.04975666, 0.04286148, 0.09360881, 0.08471966, 0.014344, 0.01221422, 0.09403581, 0.07944607, 0.02534608, 0.00833269, 0.14250581,
                      0.0836638, 0.16662271, 0.16402948, 0.11201444, 0.15168502, 0.04981251, 0.2310533, 0.18554246, 0.05107217, 0.09482361, 0.00804513, 0.0083411, 0.02594944, 0.01445219, 0.11921273, 0.02309612, 0.02547163, 0.01260222, 0.00355269, 0.14044931, 0.01781433, 0.10316608, 0.05325515, 0.33837975, 0.16515635])
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
