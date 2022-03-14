import functools
from typing import List, Tuple

from ray.rllib import SampleBatch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor


def lazy_tensor_dict(postprocessed_batch: SampleBatch, device=None):
    if not isinstance(postprocessed_batch, SampleBatch):
        postprocessed_batch = SampleBatch(postprocessed_batch)
    postprocessed_batch.set_get_interceptor(
        functools.partial(convert_to_torch_tensor, device=device)
    )
    return postprocessed_batch


def _linear_interpolation(left, right, alpha):
    return left + alpha * (right - left)


# Learning Rate Scheduling
# https://d2l.ai/chapter_optimization/lr-scheduler.html
class PiecewiseSchedule:
    def __init__(
        self,
        endpoints: List[Tuple[int, float]],
        framework: str,
        interpolation=_linear_interpolation,
        outside_value=None,
    ):
        """
        Args:
            endpoints (List[Tuple[int,float]]): A list of tuples
                `(t, value)` such that the output is an interpolation (given by the `interpolation` callable)
                between two values.

                E.g.
                    t = 400 and endpoints=[(0, 20.0), (500, 30.0)]
                output=20.0 + 0.8 * (30.0 - 20.0) = 28.0

                NOTE: All the values for time must be sorted in an increasing order.

            interpolation (callable): A function that takes the left-value,
                the right-value and an alpha interpolation parameter
                (0.0=only left value, 1.0=only right value), which is the
                fraction of distance from left endpoint to right endpoint.

            outside_value (Optional[float]):
                If t in call to `value` is outside all the intervals in `endpoints` this value is returned.
                If None then an AssertionError is raised when outside value is requested.
        """
        super().__init__(framework=framework)

        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self.interpolation = interpolation
        self.outside_value = outside_value
        self.endpoints = [(int(e[0]), float(e[1])) for e in endpoints]

    def _value(self, t):
        # Find t in our list of endpoints.
        for (l_t, l), (r_t, r) in zip(self.endpoints[:-1], self.endpoints[1:]):
            # When found, return an interpolation (default: linear).
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self.interpolation(l, r, alpha)

        # t does not belong to any of the pieces, return `self.outside_value`.
        assert self.outside_value is not None
        return self.outside_value


class LearningRateSchedule:
    """Mixin for TorchPolicy that adds a learning rate schedule."""

    def __init__(self, lr, lr_schedule):
        self._lr_schedule = None
        if lr_schedule is None:
            self.cur_lr = lr
        else:
            self._lr_schedule = PiecewiseSchedule(
                lr_schedule, outside_value=lr_schedule[-1][-1], framework=None
            )
            self.cur_lr = self._lr_schedule.value(0)

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr


class EntropyCoeffSchedule:
    """Mixin for TorchPolicy that adds entropy coeff decay."""

    def __init__(self, entropy_coeff, entropy_coeff_schedule):
        self._entropy_coeff_schedule = None
        if entropy_coeff_schedule is None:
            self.entropy_coeff = entropy_coeff
        else:
            # Allows for custom schedule similar to lr_schedule format
            if isinstance(entropy_coeff_schedule, list):
                self._entropy_coeff_schedule = PiecewiseSchedule(
                    entropy_coeff_schedule,
                    outside_value=entropy_coeff_schedule[-1][-1],
                    framework=None,
                )
            else:
                # Implements previous version but enforces outside_value
                self._entropy_coeff_schedule = PiecewiseSchedule(
                    [[0, entropy_coeff], [entropy_coeff_schedule, 0.0]],
                    outside_value=0.0,
                    framework=None,
                )
            self.entropy_coeff = self._entropy_coeff_schedule.value(0)

    def on_global_var_update(self, global_vars):
        super(EntropyCoeffSchedule, self).on_global_var_update(global_vars)
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )
