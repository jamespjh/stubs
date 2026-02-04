from multiply.benchmark import benchmark_range, benchmark
import numpy as np


def spin(N):
    sum([i*i for i in range(N)])


def test_benchmark():
    assert (benchmark(spin, 10000) > benchmark(spin, 1000))


def test_benchmark_range():
    sizes = 10**np.arange(1, 4)
    results = benchmark_range(spin, sizes)
    assert results.shape == (2, sizes.shape[0])
    assert all(results[0, :] > 0)
