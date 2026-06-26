from __future__ import annotations

from numbers import Integral
from typing import Sequence


def limit_samples_bounds(limit_samples: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(limit_samples, Integral):
        return 0, int(limit_samples)

    start, stop = limit_samples
    return int(start), int(stop)


def sample_in_limit_samples(i_sample: int, limit_samples: int | Sequence[int]) -> bool:
    start, stop = limit_samples_bounds(limit_samples)
    return start <= int(i_sample) < stop
