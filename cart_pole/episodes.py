from typing import Any, List, Tuple, Union

import numpy as np
import torch
from numpy import ndarray
from ray.rllib import SampleBatch
from ray.rllib.evaluation import compute_advantages
from ray.rllib.models.modelv2 import restore_original_dimensions

from cart_pole.utils import lazy_tensor_dict


class Episode:
    def __init__(self, env, policies, percentile=70, gamma=0.99):
        self.env = env
        self.policies = policies
        self.percentile = percentile
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.episode_id = 0
        self.reward_mean = 0.0

    def reset(self):
        self.episode_id = 0

    def loader(self, batch_size):
        batch_counter = 0
        obs = self.env.reset()
        sample_batch = SampleBatch(obs=[], actions=[], rewards=[], done=[], eps_id=[])
        while True:
            action = self.policies.compute_action(SampleBatch(obs=[obs]))
            next_obs, reward, done, _ = self.env.step(action)
            sample_batch = sample_batch.concat(
                SampleBatch(
                    obs=[obs],
                    actions=[action],
                    rewards=[reward],
                    done=[done],
                    eps_id=[self.episode_id],
                )
            ).copy()

            if done:
                self.episode_id += 1
                batch_counter += 1
                next_obs = self.env.reset()
                if batch_counter == batch_size:
                    self.reward_mean = self.compute_reward_mean(sample_batch)
                    yield sample_batch
                    sample_batch = SampleBatch(
                        obs=[], actions=[], rewards=[], done=[], eps_id=[]
                    )
                    batch_counter = 0

            obs = next_obs

    def to_tensor(self, batch):
        batch_list = [
            compute_advantages(
                rollout=sample_batch,
                last_r=0.0,
                gamma=self.gamma,
                use_gae=False,
                use_critic=False,
            )
            for sample_batch in batch.split_by_episode()
        ]
        batch = self.restore(batch_list)
        _obs_batch = lazy_tensor_dict(batch, device=self.device)
        input_dict = restore_original_dimensions(
            _obs_batch, self.env.observation_space, "torch"
        )
        return input_dict

    def compute_reward_mean(self, batch):
        gs = np.array([len(b) for b in batch.split_by_episode()])
        reward_mean = np.mean(gs)
        return reward_mean

    def filter(
        self, batch: SampleBatch
    ) -> Tuple[
        Union[Union[dict, tuple], Any], Union[int, float, complex, ndarray], ndarray
    ]:
        """filter good trajectories based on total rewards.

        Parameters
        ----------
        batch

        Returns
        -------

        """
        _batch = batch.split_by_episode()
        gs = np.array([len(b) for b in _batch])
        reward_mean = np.mean(gs)
        bound = np.percentile(gs, self.percentile)

        idx = np.where(gs > bound)[0]
        _elite_batch = np.array(_batch)[idx]
        _elite_batch = self.restore(_elite_batch)
        input_dict = self.to_tensor(_elite_batch)
        return input_dict, bound, reward_mean

    def restore(self, batch_list: List[SampleBatch]) -> SampleBatch:
        batch = batch_list[0]
        if len(batch_list) == 1:
            return batch

        for i in range(1, len(batch_list)):
            batch = batch.concat(batch_list[i])
        return batch
