from typing import Dict

from ray.rllib import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance

from cart_pole.utils import lazy_tensor_dict

torch, nn = try_import_torch()


class PPOPolicy:
    def __init__(self, config):
        self.config = config
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]

    def loss(self, model, dist_class, train_batch: SampleBatch):
        """compute loss.

        Objective function is here:
            J = E [ pi(a|s) / pi_old(a|s) A_t ]

        It is calculated from clipped surrogate objective as below:

            surrogate_loss = E [ min(r_t * A_t, clip(1 - epsilon, 1 + epsilon) * A_t ) ]

        where, r_t = pi(a|s) / pi_old(a|s)

        If parameters in value function network share with policy networks,
        the loss function need to combine the policy surrogate and a value function error term as follows:

            value_loss = (A_t - V(s_t))^2

        PPO uses a general advantage estimator (GAE).
            A_t = sigma_t + (gamma * lambda) sigma_{t+1} + (gamma * lambda)^2 sigma_{t+2} + ...
            sigma_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        The sigma_t is TD error. The original A3C estimation is a special case of the proposed method with $\lambda=1$.
        Combining these terms, we obtain the following objective, which is (approximately) maximized each iteration:

            L = E[surrogate_loss - vf_coeff * value_loss + entropy_coeff * S[pi](s)]


        Parameters
        ----------
        model
        dist_class
        train_batch

        Returns
        -------

        """
        logits, _ = model(train_batch)
        curr_action_dist = dist_class(logits, model)
        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = torch.mean(action_kl)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        # compute surrogate policy loss
        surrogate_loss = torch.min(
            train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
            train_batch[Postprocessing.ADVANTAGES]
            * torch.clamp(
                logp_ratio, 1 - self.config["clip_param"], 1 + self.config["clip_param"]
            ),
        )

        # Compute a value target loss
        value = model.value_function()
        vf_loss = torch.pow(value - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss_clipped = torch.clamp(vf_loss, self.config["vf_clip_param"])

        # Regularization term (Entropy bonus to ensure sufficient exploration).
        curr_entropy = curr_action_dist.entropy()

        # Compute a total loss. An entropy_coeff is adaptive.
        total_loss = torch.mean(
            -surrogate_loss
            + self.config["vf_loss_coef"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        # Add mean_kl_loss (alread processed through mean.torch) if necessary.
        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # Store values for stats function in model (tower), such that for multi-GPU,
        # we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = torch.mean(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = torch.mean(vf_loss_clipped)
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], model.value_function()
        )
        model.tower_stats["mean_entropy"] = torch.mean(curr_entropy)
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

    def _value(self, **input_dict: Dict[str, SampleBatch]):
        """Compute GAE

        Parameters
        ----------
        input_dict

        Returns
        -------

        """
        input_dict = lazy_tensor_dict(input_dict)
        model_out, _ = self.model(input_dict)
        # [0] = remove the batch dim.
        return self.model.value_function()[0].item()

    def update_kl(self, sampled_kl):
        # update the current KL value based on the recently measured value.
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5

        return self.kl_coeff
