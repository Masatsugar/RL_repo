import gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import restore_original_dimensions

from cart_pole.episodes import Episode
from cart_pole.utils import lazy_tensor_dict


class Policy(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super(Policy, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.observation_space = observation_space
        self.action_space = action_space
        self.shared_layer = nn.Sequential(
            nn.Linear(observation_space.shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.actor = nn.Sequential(nn.Linear(32, out_features=action_space.n))
        self.critic = nn.Sequential(nn.Linear(32, out_features=1))
        self.state_value = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_dict):
        x = input_dict["obs"]
        x = self.shared_layer(x)
        logits = self.actor(x)
        self.state_value = self.critic(x)
        return logits

    def obs_to_input_dict(self, obs):
        _obs_batch = lazy_tensor_dict(obs, device=self.device)
        input_dict = restore_original_dimensions(
            _obs_batch, self.observation_space, "torch"
        )
        return input_dict

    def compute_priors_and_value(self, obs):
        input_dict = self.obs_to_input_dict(obs)
        with torch.no_grad():
            logits = self.forward(input_dict)
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = self.softmax(logits)
            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            return priors, value

    def compute_action(self, obs, **kwargs):
        input_dict = self.obs_to_input_dict(obs)
        with torch.no_grad():
            logits = self.forward(input_dict)
            probs = self.softmax(logits).cpu().numpy()[0]
            action = np.random.choice(len(probs), p=probs)
        return action


def run():
    """Cross Entropy Method: versatile Monte Carlo technique.
    https://people.smp.uq.edu.au/DirkKroese/ps/eormsCE.pdf

    """
    env = gym.make("CartPole-v1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=None,
    )
    xentropy = nn.CrossEntropyLoss()
    if device == "cuda":
        policy = policy.cuda()
        xentropy = xentropy.cuda()

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    optimizer.zero_grad()
    dataset = Episode(env, policies=policy, percentile=70)
    trainloader = dataset.loader(batch_size=16)
    for i, batch in enumerate(trainloader):
        input_dict, reward_bound, reward_mean = dataset.filter(batch)
        action_scores_v = policy(input_dict)
        loss_v = xentropy(action_scores_v, input_dict["actions"])
        loss_v.backward()
        optimizer.step()
        print(
            f"{i}: loss={loss_v.item():.3f}, reward_mean={reward_mean:.1f}, reward_bound={reward_bound:.1f}"
        )
        if reward_mean >= 200:
            print("Solved")
            break

    return policy


if __name__ == "__main__":
    policy = run()
    env = gym.make("CartPole-v1")
    obs = env.reset()
    total_reward = 0.0
    while True:
        action = policy.compute_action(SampleBatch(obs=[obs]))
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            print(f"total reward: {total_reward}")
            break
        obs = next_obs
