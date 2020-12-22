import random
import numpy as np
import pandas as pd
import numpy.random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import EpsilonGreedy, UCB1, ThompsonSampling
from utils import BernoulliArm
from testdata import y


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

    # 問題設定: 腕：100本のうち、あたりは1つとする
    theta = np.array([0.011964127231654534, 0.019337503108103803, 0.015688672024740072, 0.9393156898855956, 0.006808640353145985, 0.19286894167378513, 0.02132537389757775, 0.06851518549926339, 0.0699371968605012, 0.05666969943123235, 0.01853045755544196, 0.017665506418041335, 0.023365250995323654, 0.034503162711730456, 0.004654767688494326, 0.1326408027422968, 0.01746609851873785, 0.05444285322915503, 0.0002287146168246802, 0.42719445134966544, 0.09275261220767882, 0.04973736330795265, 0.4019212024032282, 0.26994664135552365, 0.026383804742013194, 0.11412866680829468, 0.19420952486068255, 0.4906082637882126, 0.43446786907477103, 0.18335319336862918, 0.022384556752654685, 0.036764326342966926, 0.03728038558607678, 0.03233119667090953, 0.03942664432338577, 0.15043428209146537, 0.011635391938107531, 0.017397392973402443, 0.18117390644195594, 0.02147976787966694, 0.03856702916795387, 0.020557298282393194, 0.11096026396352407, 0.07613980818888658, 0.3058863478072673, 0.007427407902581744, 0.10275766354765908, 0.06253970457236381, 0.06834157599658969, 0.2164078549428506,
                      0.11037323091934008, 0.026501570853083033, 0.006775741395116116, 0.11400869338765442, 0.005315869652356669, 0.028466122883488258, 0.031866889138019376, 0.03921143639110928, 0.03474125392978029, 0.16578248703330406, 0.01572715800552972, 0.03291197010427882, 0.0243747647635273, 0.08130038394411689, 0.01842953361767227, 0.09588500489596392, 0.02467293027336295, 0.07268646280474562, 0.07445927738235025, 0.030788204787231457, 0.21590219067678815, 0.0033504997256582725, 0.1773014173562062, 0.08893962416547989, 0.008945159979398072, 0.5526220868821929, 0.00783142012251489, 0.08648698713535945, 0.050961361270139144, 0.07442860733814968, 0.03982213782840011, 0.1333491834737458, 0.226385008086726, 0.29084344593724526, 0.15884969458609843, 0.010311125480698402, 0.28118299777676026, 0.12316144685550937, 0.053036564329083144, 0.20592667716474716, 0.004849084056489814, 0.031096446110345656, 0.1593056015308471, 0.12048721234279149, 0.12590833723921918, 0.2701889293458395, 0.09241861796690845, 0.03159262586536227, 0.02818685225597747, 0.1833511581561177])
    n_arms = len(theta)
    # random.shuffle(theta)
    print(theta)

    arms = map(lambda x: BernoulliArm(x), theta)
    arms = list(arms)

  # EpsilonGreedy0.01のepsilonの値設定
    algos = {'Eps001': EpsilonGreedy([], [], epsilon=0.01), 'Eps005': EpsilonGreedy([], [], epsilon=0.05), 'Eps01': EpsilonGreedy([], [], epsilon=0.1),  'UCB': UCB1(
        [], []), 'TS': ThompsonSampling([], [])}
    for key, algo in algos.items():
        run(algo, label=key)

    plt.show()
