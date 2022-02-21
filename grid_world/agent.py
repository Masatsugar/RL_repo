import random
from collections import Counter, defaultdict


class Agent:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.transitions = defaultdict(Counter)  # (state, action)
        self.reward_table = defaultdict(float)  # (state, action, next_state)
        self.value_table = defaultdict(float)

    def play_episode(self, is_random=False):
        total_reward = 0.0
        state = self.env.reset()
        while True:
            action = (
                self.env.action_space.sample()
                if is_random
                else self.compute_action(state)
            )
            next_state, reward, is_done, _ = self.env.step(action)
            self.set_state_action(state, action, next_state, reward)
            total_reward += reward
            if is_done:
                break
            state = next_state
        return total_reward

    def play_random_steps(self, n=100, is_random=True):
        for _ in range(n):
            self.play_episode(is_random=is_random)

    def compute_action(self, state):
        raise NotImplementedError

    def set_state_action(self, state, action, next_state, reward):
        self.reward_table[state, action, next_state] = reward
        self.transitions[state, action][next_state] += 1


class Vlearning(Agent):
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        super(Vlearning, self).__init__(env, gamma, epsilon)
        self.env = env
        self.state = self.env.reset()

    def compute_action(self, state):
        if random.random() > self.epsilon:
            best_action, best_value = None, None
            for action in range(self.env.action_space.n):
                action_value = self.compute_action_value(state, action)
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            return best_action
        else:
            return self.env.action_space.sample()

    def compute_action_value(self, state, action):
        target_counts = self.transitions[state, action]
        total = sum(target_counts.values())
        action_value = 0.0
        for target_state, count in target_counts.items():
            reward = self.reward_table[state, action, target_state]
            action_value += (count / total) * (
                reward + self.gamma * self.value_table[target_state]
            )
        return action_value

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [
                self.compute_action_value(state, action)
                for action in range(self.env.action_space.n)
            ]
            self.value_table[state] = max(state_values)


class Qlearning(Agent):
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        super(Qlearning, self).__init__(env, gamma, epsilon)
        self.env = env
        self.state = self.env.reset()

    def compute_action(self, state):
        if random.random() > self.epsilon:
            best_action, best_value = None, None
            for action in range(self.env.action_space.n):
                action_value = self.value_table[state, action]  # select max Q-value
                if best_value is None or best_value < action_value:
                    best_value = action_value
                    best_action = action
            return best_action
        else:
            return self.env.action_space.sample()

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transitions[state, action]
                total = sum(target_counts.values())
                for target_state, count in target_counts.items():
                    reward = self.reward_table[state, action, target_state]
                    best_action = self.compute_action(target_state)
                    action_value += (count / total) * (
                        reward
                        + self.gamma * self.value_table[target_state, best_action]
                    )
                    self.value_table[state, action] = action_value
