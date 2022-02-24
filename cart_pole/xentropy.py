import functools
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.torch_utils import convert_to_torch_tensor


def _lazy_tensor_dict(postprocessed_batch: SampleBatch, device=None):
    if not isinstance(postprocessed_batch, SampleBatch):
        postprocessed_batch = SampleBatch(postprocessed_batch)
    postprocessed_batch.set_get_interceptor(
        functools.partial(convert_to_torch_tensor, device=device)
    )
    return postprocessed_batch


def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor


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

    def compute_priors_and_value(self, obs_batch):
        _obs_batch = _lazy_tensor_dict(obs_batch, device=self.device)
        input_dict = restore_original_dimensions(
            _obs_batch, self.observation_space, "torch"
        )
        with torch.no_grad():
            model_out = self.forward(input_dict)
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = self.softmax(logits)
            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            return priors, value

    def compute_action(self, obs, **kwargs):
        _obs_batch = _lazy_tensor_dict(obs, self.device)
        input_dict = restore_original_dimensions(
            _obs_batch, self.observation_space, "torch"
        )
        with torch.no_grad():
            logits = self.forward(input_dict)
            probs = self.softmax(logits).cpu().numpy()[0]
            action = np.random.choice(len(probs), p=probs)
        return action


class EnvData:
    def __init__(self, env, percentile=70):
        self.env = env
        self.percentile = percentile
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def loader(self, policy, batch_size):
        eps_id = 0
        batch_counter = 0
        obs = self.env.reset()
        sample_batch = SampleBatch(obs=[], actions=[], rewards=[], done=[], eps_id=[])
        while True:
            action = policy.compute_action(SampleBatch(obs=[obs]))
            next_obs, reward, done, _ = self.env.step(action)
            sample_batch = sample_batch.concat(
                SampleBatch(
                    obs=[obs],
                    actions=[action],
                    rewards=[reward],
                    done=[done],
                    eps_id=[eps_id],
                )
            ).copy()

            if done:
                eps_id += 1
                batch_counter += 1
                next_obs = self.env.reset()
                if batch_counter == batch_size:
                    yield sample_batch
                    sample_batch = SampleBatch(
                        obs=[], actions=[], rewards=[], done=[], eps_id=[]
                    )
                    batch_counter = 0

            obs = next_obs

    def filter(self, batch):
        batch = batch.split_by_episode()
        gs = np.array([len(b) for b in batch])
        bound = np.percentile(gs, self.percentile)
        reward_mean = np.mean(gs)
        idx = np.where(gs > bound)[0]
        _elite_batch = np.array(batch)[idx]
        elite_batch = self.restore(_elite_batch)
        _obs_batch = _lazy_tensor_dict(elite_batch, device=self.device)
        input_dict = restore_original_dimensions(
            _obs_batch, self.env.observation_space, "torch"
        )
        return input_dict, bound, reward_mean

    def restore(self, batch_list: List[SampleBatch]):
        batch = batch_list[0]
        for i in range(1, len(batch_list)):
            batch = batch.concat(batch_list[i])
        return batch


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
    xentropy = nn.CrossEntropyLoss().cuda()
    if device == "cuda":
        policy = policy.cuda()
        xentropy = xentropy.cuda()

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    optimizer.zero_grad()
    dataset = EnvData(env, percentile=70)
    trainloader = dataset.loader(policy, batch_size=64)
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
