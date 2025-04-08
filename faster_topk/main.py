import numpy as np

from utils import measure_exec_time


def naive_topk(arr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(-arr, axis=-1)[..., :k]
    return np.take_along_axis(arr, indices, axis=-1), indices


def faster_topk(arr: np.ndarray, k: int):
    partition_indices = np.argpartition(arr, -k, axis=-1)[..., -k:]
    partition_values = np.take_along_axis(arr, partition_indices, axis=-1)
    sort_order = np.argsort(partition_values, axis=-1)[..., ::-1]
    topk_values = np.take_along_axis(partition_values, sort_order, axis=-1)
    topk_indices = np.take_along_axis(partition_indices, sort_order, axis=-1)

    return topk_values, topk_indices


if __name__ == "__main__":
    arr = np.random.randn(100, 1000)
    k = 20
    print("Naive topk")
    measure_exec_time(naive_topk, arr, k, verbose=True)
    print("Faster topk")
    measure_exec_time(faster_topk, arr, k, verbose=True)
