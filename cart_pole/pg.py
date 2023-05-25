import gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import restore_original_dimensions
from torch.distributions.categorical import Categorical

from cart_pole.episodes import Episode
from cart_pole.utils import lazy_tensor_dict

FAKE_BATCH = SampleBatch(
    {
        SampleBatch.OBS: np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            dtype=np.float32,
        ),
        SampleBatch.ACTIONS: np.array([0, 1, 1]),
        SampleBatch.PREV_ACTIONS: np.array([0, 1, 1]),
        SampleBatch.REWARDS: np.array([1.0, -1.0, 0.5], dtype=np.float32),
        SampleBatch.PREV_REWARDS: np.array([1.0, -1.0, 0.5], dtype=np.float32),
        SampleBatch.DONES: np.array([False, False, True]),
        SampleBatch.VF_PREDS: np.array([0.5, 0.6, 0.7], dtype=np.float32),
        SampleBatch.ACTION_DIST_INPUTS: np.array(
            [[-2.0, 0.5], [-3.0, -0.3], [-0.1, 2.5]], dtype=np.float32
        ),
        SampleBatch.ACTION_LOGP: np.array([-0.5, -0.1, -0.2], dtype=np.float32),
        SampleBatch.EPS_ID: np.array([0, 0, 0]),
        SampleBatch.AGENT_INDEX: np.array([0, 0, 0]),
    }
)


class Policy(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super(Policy, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.observation_space = observation_space
        self.action_space = action_space
        self.shared_layer = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(64, out_features=action_space.n))
        self.critic = nn.Sequential(nn.Linear(64, out_features=1))
        self.state_value = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_dict):
        x = input_dict["obs"]
        x = self.shared_layer(x)
        logits = self.actor(x)
        self.state_value = self.critic(x)
        return logits

    def value_function(self):
        return self.state_value

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
    env = gym.make("CartPole-v1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=None,
    )
    loss_func = nn.MSELoss()
    if device == "cuda":
        policy = policy.cuda()
        loss_func = loss_func.cuda()

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    optimizer.zero_grad()
    episode = Episode(env, policies=policy)
    trainloader = episode.loader(batch_size=32)
    for i, train_batch in enumerate(trainloader):
        # input_dict, _, _ = episode.filter(train_batch)
        input_dict = episode.to_tensor(train_batch)  # , use_gae=True, use_critic=True)
        logits = policy(input_dict)
        probs = policy.softmax(logits)
        action_dist = Categorical(probs)
        state_value = policy.value_function().flatten()

        # L = -E[ log(pi(a|s)) * A]
        log_probs = action_dist.log_prob(input_dict[SampleBatch.ACTIONS])

        # REINFORCE
        policy_loss = -torch.mean(log_probs * input_dict[Postprocessing.ADVANTAGES])

        # Actor loss
        # policy_loss = -torch.mean(log_probs * state_value)

        # Critic Loss
        # value_loss = loss_func(state_value, input_dict[Postprocessing.ADVANTAGES])

        # Entropy (regluarization)

        # loss = policy_loss + value_loss

        policy_loss.backward()
        optimizer.step()
        print(
            f"{i}: loss={policy_loss.item():.3f}, reward_mean={episode.reward_mean:.1f}"
        )
        if episode.reward_mean >= 400:
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
