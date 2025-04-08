import random
import statistics
import time
from typing import Callable

import numpy as np
import torch


def measure_exec_time(
    func: Callable, *args, n: int = 100, verbose: bool = False
) -> tuple[float, float]:
    times = []
    for _ in range(n):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if n > 1 else 0.0
    if verbose:
        print(f"Mean: {mean_ms:.3f} ms, Std: {std_ms:.3f} ms")

    return mean_ms, std_ms


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
