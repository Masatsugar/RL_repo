import gym

from grid_world.env import CustomMaze
from grid_world.agent import Vlearning, Qlearning

if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    # env = CustomMaze()
    # agent = Vlearning(env, gamma=0.90, epsilon=0.0)
    agent = Qlearning(env)
    iter_n = 0
    best_reward = 0.0
    test_episode = 10
    while True:
        iter_n += 1
        agent.play_random_steps(100)
        for _ in range(env.observation_space.n):
            agent.value_iteration()
        reward = 0.0
        for _ in range(test_episode):
            reward += agent.play_episode()
        reward /= test_episode

        if reward > best_reward:
            print(f"Best reward {best_reward:.3f} -> {reward:.3f}")
            best_reward = reward
        if reward > 0.8:
            print(f"Solved: {iter_n} iterations. Best reward: {best_reward}.")
            break
