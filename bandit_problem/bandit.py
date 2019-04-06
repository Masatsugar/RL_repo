import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ベルヌーイ分布に従う問題
class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        randp = random.random()
        # print("Your Trial {:.4f}".format(randp))
        if randp > self.p:
            return 0.0
        else:
            return 1.0


class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts  # armの引く回数
        self.values = values  # 引いたarmから得られた報酬の平均値

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if random.random() > self.epsilon:
            # 確率epsilonで探索（アームを選ぶ）[0, アームの総数)
            # print("Explore: values {}".format(self.values))
            # return np.random.randint(0, len(self.values))
            return np.argmax(self.values)
        else:
            # 確率1- epsilonで活用（報酬が最大となっているアームを選ぶ）
            # print("Exploit: values {}\n".format(self.values))
            # print("Number {}".format(np.argmax(self.values)))
            return random.randrange(len(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]  # 今回のアームを選択した回数
        value = self.values[chosen_arm]  # 更新前の平均報酬額
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        # print("n {}".format(n))
        self.values[chosen_arm] = new_value


class UCB1():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        for arm in range(n_arms):
            bonus = np.sqrt((2 * np.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


# test_algorithm()メソッドの定義
def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    rewards = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    sim_nums = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)

    for sim in range(num_sims):
        if sim % 200 == 0:
            # print("============================\n")
            print("SIM : {}".format(sim))
            # print("\n============================\n")

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


def categorical_draw(probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i

    return len(probs) - 1


class PolicyGradient:
    def __init__(self, counts, values, n_arms):
        self.counts = counts
        self.values = values
        self.n_arms = n_arms

        self.total_values = [0 for _ in range(self.n_arms)]
        self.total_counts = 1
        self.episodes = 0

        # soft max parameter
        self.theta = np.ones(self.n_arms) + np.random.normal(0, 0.01, self.n_arms)

        # policy gradient
        self.grad_reward = 0
        self.mean_grad_reward = 0

        # hyper parameter
        self.beta = 1.0
        self.eta = 0.1

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

        # M : episodes T : total step
        # (1/M)∑(1/T)∑[∇log(π(a))* rewards]
        self.total_counts = np.sum(self.counts)
        if self.total_counts == 0:
            self.total_counts = 1
        self.mean_grad_reward = self.grad_reward / self.total_counts
        self.episodes += 1

        # update theta
        if self.episodes % 10 == 0:
            self.theta = self.reinforce(self.theta)
            print("EPISODE {} : Update Policy Probability {}".format(self.episodes, self.soft_max(self.theta)))
        # print("(1/T)∑∇logπ(a)R = {}\n".format(self.mean_grad_reward))

    def select_arm(self):
        # t = sum(self.counts) + 1
        # beta= 1 / np.log(t + 0.0000001)
        beta = 1.0
        logits = beta * self.theta
        probs = self.soft_max(logits)
        return categorical_draw(probs)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]  # 今回のアームを選択した回数
        value = self.values[chosen_arm]  # 更新前の平均報酬額
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        self.grad_reward += self.grad_ln_pi(self.theta).dot(self.values)  # new_value

    def reinforce(self, theta):
        # REINFORCE
        nabla_J = self.mean_grad_reward / self.episodes
        new_theta = theta + self.eta * nabla_J
        self.mean_grad_reward += self.mean_grad_reward
        return new_theta

    @staticmethod
    def soft_max(logits):
        pi = [np.exp(var) / np.nansum(np.exp(logits)) for var in logits]
        pi = np.nan_to_num(pi)
        return pi

    def grad_soft_max(self):
        pi = self.soft_max(self.theta)
        dpi = []
        for i in pi:
            for j in pi:
                if i == j:
                    dpi.append(i * (1 - i))
                else:
                    dpi.append(-i * j)
        grad_pi = np.array(dpi).reshape(len(pi), -1)
        return grad_pi

    def grad_ln_pi(self):
        pi = self.soft_max(self.theta)
        dlog_pi = []
        for i in pi:
            for j in pi:
                if i == j:
                    dlog_pi.append(1 - i)
                else:
                    dlog_pi.append(-j)
        dlog_pi = np.array(dlog_pi).reshape(len(pi), -1)
        return dlog_pi


# ------------------------------------------------------------------------------
# MAIN 
# ------------------------------------------------------------------------------
def main():
    theta = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
    n_arms = len(theta)
    random.shuffle(theta)

    arms = map(lambda x: BernoulliArm(x), theta)
    arms = list(arms)

    for epsilon in [0, 0.1, 0.2, 0.3]:
        print("epsilon = {}".format(epsilon))
        algo = EpsilonGreedy(epsilon, [], [])
        algo.initialize(n_arms)
        results = test_algorithm(algo, arms, num_sims=2000, horizon=200)

        df = pd.DataFrame({"times": results[1], "rewards": results[3]})
        grouped = df["rewards"].groupby(df["times"])

        plt.plot(grouped.mean(), label="epsilon=" + str(epsilon))

    print('UCB1')
    algo = UCB1([], [])
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sims=2000, horizon=200)
    df = pd.DataFrame({"times": results[1], "rewards": results[3]})
    grouped = df["rewards"].groupby(df["times"])
    plt.plot(grouped.mean(), label="UCB1")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()
