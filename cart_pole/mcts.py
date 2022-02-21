"""
Mcts implementation modified from
https://github.com/brilee/python_uct/blob/master/numpy_impl.py
"""
from collections import defaultdict
from typing import Tuple

import gym
import numpy as np
from tqdm import tqdm

from cart_pole.env import CartPole
from cart_pole.q_learning import QTable


class Node:
    def __init__(self, action, obs, done, reward, state, mcts, parent=None):
        self.env = parent.env
        self.action = action
        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.action_space_size = self.env.action_space.n

        self.child_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # N

        self.reward = reward
        self.done = done
        self.state = state
        self.obs = obs

        self.mcts = mcts

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(N={self.child_number_visits}, "
            f"Q={self.child_total_value}, U={self.child_U()})"
        )

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.action]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.action] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return (
            np.sqrt(self.number_visits)
            * self.child_priors
            / (1 + self.child_number_visits)
        )

    def best_action(self):
        return np.argmax(self.child_Q() + self.mcts.c_puct * self.child_U())

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            self.env.set_state(self.state)
            obs, reward, done, _ = self.env.step(action)
            next_state = self.env.get_state()
            self.children[action] = Node(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=done,
                obs=obs,
                mcts=self.mcts,
            )
        return self.children[action]

    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1  # N
            current.total_value += value  # Q
            current = current.parent


class RootParentNode:
    def __init__(self, env):
        self.parent = None
        self.child_total_value = defaultdict(float)
        self.child_number_visits = defaultdict(float)
        self.env = env


class MCTS:
    def __init__(self, model, num_sims=100, c_puct=1.0, exploit=False):
        self.model = model
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.exploit = exploit
        self.actions = []

    def compute_action(self, node):
        for _ in range(self.num_sims):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.model.compute_priors_and_value(leaf)
                leaf.expand(child_priors)
            leaf.backup(value)

        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.sum(tree_policy)

        if self.exploit:
            action = np.argmax(tree_policy)
        else:
            action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.children[action]


class Policy:
    def __init__(self, env, num_digit=6):
        self.env = env
        self.q_table = QTable(env, num_digit)

    def compute_priors_and_value(self, node: Node) -> Tuple[np.ndarray, float]:
        state = self.q_table.digitize(node.obs)
        self.update(state, node)
        value = max(self.q_table[state][:])
        priors = np.ones(self.env.action_space.n)
        priors /= sum(priors)
        return priors, value

    def update(self, state, node):
        if node.action:
            action_value = node.state[2] / (1 + node.child_number_visits[node.action])
            self.q_table[state, action] = action_value
            if action_value > 0:
                print(action_value, state, action)


if __name__ == "__main__":
    env = CartPole()
    env.reset()
    parent = RootParentNode(env)
    policy = Policy(env)
    mcts = MCTS(policy, num_sims=10000)
    node = Node(
        state=policy.env.get_state(),
        obs=policy.env.reset(),
        reward=0,
        done=False,
        action=None,
        parent=RootParentNode(env=policy.env),
        mcts=mcts,
    )

    # child_priors, value = policy.compute_priors_and_value(leaf)
    tree_policy, action, children = mcts.compute_action(node)
