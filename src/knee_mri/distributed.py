"""Multi-GPU (DistributedDataParallel) helpers for one training run.

`torchrun --nproc_per_node=N` starts N processes and sets WORLD_SIZE, RANK and LOCAL_RANK;
without them every helper here is a no-op and the run is the ordinary single-device one.

Each rank trains on every world_size-th study of the same seeded order (ShardSampler) with
accumulation_steps / world_size local micro-batches per window, so the k-th global window
holds exactly the studies the single-device run puts in its k-th window, and the optimizer
step (gradients all-reduced at the window's last backward) is the same step.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

# Validation, checkpoint writing and the epoch-boundary loader respawn run while the other
# ranks wait in a collective; the default 10 minutes is too tight for a large validation fold.
PROCESS_GROUP_TIMEOUT = datetime.timedelta(hours=2)


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def init_distributed() -> None:
    """Join the process group when launched by torchrun with more than one process."""
    if env_world_size() <= 1 or is_distributed():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        if torch.cuda.device_count() <= local_rank:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible "
                "(check CUDA_VISIBLE_DEVICES against --nproc_per_node)"
            )
        torch.cuda.set_device(local_rank)
        backend = "nccl" if dist.is_nccl_available() else "gloo"  # Windows builds have no NCCL
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, timeout=PROCESS_GROUP_TIMEOUT)


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0")) if is_distributed() else 0


def is_main() -> bool:
    return rank() == 0


def device_str() -> str | None:
    """This rank's device, or None for the single-device default (select_device decides)."""
    if not is_distributed() or not torch.cuda.is_available():
        return None
    return f"cuda:{local_rank()}"


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Rank `src`'s object on every rank (pickled; for small Python values)."""
    if not is_distributed():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def all_gather_object(obj: Any) -> list[Any]:
    """Every rank's object, in rank order (pickled; for small Python values)."""
    if not is_distributed():
        return [obj]
    gathered: list[Any] = [None] * world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def _reduce_device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")


def all_reduce_any(flag: bool) -> bool:
    """True on every rank when it is true on any rank."""
    if not is_distributed():
        return bool(flag)
    value = torch.tensor([1 if flag else 0], dtype=torch.int64, device=_reduce_device())
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def all_reduce_sum(values: np.ndarray | float | int) -> np.ndarray:
    """Element-wise sum over the ranks, as float64 (counts stay exact below 2**53)."""
    array = np.asarray(values, dtype=np.float64)
    if not is_distributed():
        return array
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(_reduce_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


@torch.no_grad()
def sync_buffers_(module: torch.nn.Module) -> None:
    """Give every rank the same buffers: float ones (BatchNorm running statistics) averaged over
    the ranks, the others (num_batches_tracked) taken from rank 0. In place; a no-op on buffers
    that already agree (the average of equal values is exact)."""
    if not is_distributed():
        return
    for buffer in module.buffers():
        if buffer.is_floating_point():
            dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
            buffer.div_(world_size())
        else:
            dist.broadcast(buffer, src=0)


class ShardSampler(Sampler):
    """This rank's share of the wrapped sampler's order: positions rank, rank + W, rank + 2W, ...

    Every rank wraps an identically seeded sampler, so all ranks see the same order. It is
    padded from its own start to a multiple of the world size (at most W-1 repeated studies
    per epoch), so every rank yields the same number of indices and the collectives line up.
    """

    def __init__(self, sampler: Sampler, rank: int, world_size: int) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        self.sampler = sampler
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        # Lazily, like the single-device loader: the wrapped iterator is only run to its end
        # (StopIteration) once this one is. A seeded RandomSampler draws one more permutation
        # from its generator at that point, so draining it eagerly would shift every later
        # epoch's order away from the single-device run's.
        n = len(self.sampler)
        if n == 0:
            return
        inner = iter(self.sampler)
        head: list[int] = []  # the first studies, reused as padding
        for position in range(len(self) * self.world_size):
            if position < n:
                index = int(next(inner))
                if len(head) < self.world_size:
                    head.append(index)
            else:
                index = head[(position - n) % len(head)]
            if position % self.world_size == self.rank:
                yield index
        for _ in inner:  # reached only when the consumer runs this iterator to its end
            pass

    def __len__(self) -> int:
        return -(-len(self.sampler) // self.world_size)
