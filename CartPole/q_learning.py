import numpy as np
import gym
from queue import Queue


class QTable:
    def __init__(self, num_digit, num_state, num_action):
        """
        cart_x = (-2.4, 2.4)
        cart_v = (-Inf, Inf)
        pole_angle = (-41.8°, 41.8)
        pole_v = (-Inf, Inf)
        """
        self.q_table = np.random.uniform(low=0, high=1, size=(num_digit ** num_state, num_action))
        self.num_digit = num_digit
        self.num_action = num_action
        self.bound = np.array([[-2.4, 2.4],
                               [-3., 3.],
                               [-.5, .5],
                               [-2., 2.]])

    @property
    def num_digit(self):
        return self.__num_digit

    @num_digit.setter
    def num_digit(self, digit: int):
        self.__num_digit = digit

    # digitizes the continuous state into discrete state
    def digitize(self, observation: np.ndarray) -> int:

        bins = lambda x, y, z: np.linspace(x, y, z + 1)[1:-1]
        bins_list = list(map(lambda x, y: bins(x, y, self.num_digit), self.bound[:, 0],self.bound[:, 1]))
        # bins_list = [bins(x, y, self.num.digit) for x, y in zip(bound[:, 0], bound[,:1])]
        digit = [np.digitize(obs, lst) for obs, lst in zip(observation, bins_list)]

        return sum([dig * (self.num_digit ** i) for i, dig in enumerate(digit)])

    def update(self, observation, action, reward, observation_next):
        return

    def choose_action(self, observation, episode: int):
        return


class Action:
    def __init__(self, num_action):
        self.num_action = num_action
        self.episode = 0

    def greedy(self, q_table, state, ) -> int:
        return np.argmax(q_table[state][:])

    def epsilon_greedy(self, q_table, state, episode) -> int:
        self.episode = episode
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon < np.random.uniform(0, 1):
            action = np.argmax(q_table[state][:])
        else:
            action = np.random.choice(self.num_action)
        return action


class QLearning(QTable):
    def __init__(self, num_digit, num_state, num_action, eta=0.5, gamma=0.99):
        super().__init__(num_digit, num_state, num_action)
        self.action = Action(self.num_action)
        self.eta = eta
        self.gamma = gamma

    def update(self, observation, action, reward, observation_next):
        state = self.digitize(observation)
        state_next = self.digitize(observation_next)

        # max_a Q(s_{t+1}, a_t)
        # Off-policy
        max_q = max(self.q_table[state_next][:])

        # Q(s_t, a_t) := Q(s_t, a_t) + eta * ( reward + gamma * max Q(s_{t+1}, a_t) - Q(s_t, a_t))
        self.q_table[state, action] = \
            self.q_table[state, action] + \
            self.eta * (reward + self.gamma * max_q - self.q_table[state, action])

    def choose_action(self, observation, episode: int) -> int:
        state = self.digitize(observation)
        return self.action.epsilon_greedy(q_table=self.q_table, state=state, episode=episode)


class Sarsa(QLearning):
    def __init__(self, num_digit, num_state, num_action, eta=0.5, gamma=0.99):
        super().__init__(num_digit, num_state, num_action, eta, gamma)

    def update(self, observation, action, reward, observation_next):
        state = self.digitize(observation)
        state_next = self.digitize(observation_next)

        # A difference between Q-learning and SARSA is to use Q(s_t, a_t) instead of using max Q(s_t, a_t)
        # On-policy
        action_next = self.action.epsilon_greedy(self.q_table, state_next, self.action.episode)
        next_q = self.q_table[state_next, action_next]

        # Q(s_t, a_t) := Q(s_t, a_t) + eta * ( reward + gamma * Q(s_{t+1}, a_{t+1} - Q(s_t, a_t))
        self.q_table[state, action] = \
            self.q_table[state, action] + \
            self.eta * (reward + self.gamma * next_q - self.q_table[state, action])


class Agent:
    def __init__(self, env, episode=500, step=200, eta=0.5, gamma=0.99):
        self.env = env
        self.ETA = eta
        self.GAMMA = gamma
        self.EPISODE = episode
        self.STEP = step
        self.i_digit = 6

        self.global_moving_average_reward = 0
        self.res_queue = Queue()

    # defines some reward function.
    def Reward(self, done: bool, step: int, complete_episodes: int) -> (int, int):
        if done:
            if step < 195:
                reward = -1
                complete_episodes = 0
            else:
                reward = 1
                complete_episodes += 1
        else:
            reward = 0

        return reward, complete_episodes

    @staticmethod
    def record(episode, episode_reward, global_ep_reward, result_queue, total_loss, num_steps):
        if global_ep_reward == 0:
            global_ep_reward = episode_reward
        else:
            global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
        print(
            f"Episode: {episode} | "
            f"Moving Average Reward: {int(global_ep_reward)} | "
            f"Episode Reward: {int(episode_reward)} | "
            f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
            f"Steps: {num_steps} | "
        )
        result_queue.put(global_ep_reward)
        return global_ep_reward

    def run(self, alg: object):
        # object of each algorithm
        q = alg
        reward_avg = 0
        complete_episodes = 0
        for episode in range(self.EPISODE):
            observation = self.env.reset()
            reward_sum = 0.0
            for step in range(self.STEP):
                action = q.choose_action(observation, episode=episode)
                observation_next, reward, done, _ = self.env.step(action)

                # takes an immediate reward
                my_reward, complete_episodes = self.Reward(done, step, complete_episodes)

                # updates Q-table
                q.update(observation, action, my_reward, observation_next)

                # updates an observation
                observation = observation_next
                if done:
                    print('{0} Episode : Finished after {1} time steps'.format(episode, step + 1))
                    break

                self.global_moving_average_reward = self.record(
                    episode=episode,
                    episode_reward=reward_sum,
                    global_ep_reward=self.global_moving_average_reward,
                    result_queue=self.res_queue,
                    total_loss=0,
                    num_steps=step + 1)

                reward_sum += reward
                reward_avg += reward_sum
                final_avg = reward_avg / float(self.EPISODE)
                print("Average score across {} episodes: {}".format(self.EPISODE, final_avg))

            if complete_episodes >= 5:
                print("5 times successes")


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    i_state = env.observation_space.shape[0]
    i_action = env.action_space.n
    i_digit = 6

    alg1 = QLearning(i_digit, i_state, i_action, eta=0.5, gamma=0.99)
    alg2 = Sarsa(i_digit, i_state, i_action, eta=0.5, gamma=0.99)
    agent = Agent(env, episode=200)
    agent.run(alg2)
