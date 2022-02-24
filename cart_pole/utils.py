import functools

from ray.rllib import SampleBatch
from ray.rllib.utils.torch_utils import convert_to_torch_tensor


def lazy_tensor_dict(postprocessed_batch: SampleBatch, device=None):
    if not isinstance(postprocessed_batch, SampleBatch):
        postprocessed_batch = SampleBatch(postprocessed_batch)
    postprocessed_batch.set_get_interceptor(
        functools.partial(convert_to_torch_tensor, device=device)
    )
    return postprocessed_batch
