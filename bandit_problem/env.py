from typing import List

from matplotlib import pyplot as plt

from bandit_problem.utils import UCB1, BernoulliArm
from grid_world.agent import MonteCarlo


class MultiArmedBandit:
    def __init__(self, arms: List[BernoulliArm], max_step: int = 200):
        self.arms = arms
        self.max_step = max_step
        self.env_step = 0
        self.cumulative_reward = 0
        self.state = None
        self.trajectory = []

    def reset(self):
        self.env_step = 0
        self.cumulative_reward = 0
        return self.state

    def step(self, action):
        self.env_step += 1
        reward = self.reward(action)
        self.cumulative_reward += reward
        self.state = action
        done = True if self.env_step > self.max_step else False
        return self.state, reward, done, {}

    def reward(self, action):
        return self.arms[action].draw()

    def run(self, policy):
        self.reset()
        policy.reset()
        for i in range(self.max_step):
            action = policy.select_arm()
            _, reward, done, _ = self.step(action)
            self.trajectory.append(action)
            policy.update(action, reward)
            if done:
                break
        print(f"done: reward mean={self.cumulative_reward / self.max_step}")
        return self


if __name__ == "__main__":
    # Another Example
    NUM_SIMS = 200
    HORIZON = 200

    # 問題設定: 腕：K本のうち、あたりは1つ (0.9)とする。
    theta = [0.1, 0.1, 0.4, 0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 0.2, 0.1]
    print(theta)
    arms = list(map(lambda x: BernoulliArm(x), theta))

    env = MultiArmedBandit(arms=arms, max_step=HORIZON)
    env.reset()
    policy = UCB1()  # EpsilonGreedy(epsilon=0.2)
    env.run(policy=policy)
    print(f"Counts of chosen arms={policy.counts}, \nvalues={policy.values}")

    plt.plot(range(0, HORIZON), env.trajectory, "x-")
    plt.title(policy.__class__.__name__)
    plt.xlabel("step")
    plt.ylabel("arm")
    plt.show()
