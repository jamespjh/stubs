from multiply.benchmark import benchmark_range, benchmark
from multiply.multiply import detect_cuda, detect_jax
import numpy as np


def spin(N):
    sum([i * i for i in range(N)])


def test_benchmark():
    assert (benchmark(spin, 10000) > benchmark(spin, 1000))


def test_benchmark_range():
    sizes = 10**np.arange(1, 4)
    results = benchmark_range(spin, sizes, None)
    assert results.shape == (2, sizes.shape[0])
    assert all(results[0, :] > 0)


def test_timer():
    from multiply.benchmark import Timer
    timer = Timer(warmup=1, repeat=3)
    time = timer.timeit(spin, 10000)
    assert time > 0


def t_timer_engine(engine):
    from multiply.benchmark import Timer
    timer = Timer(warmup=1, repeat=3)
    time = timer.timeit_engine(spin, engine, 10000)
    assert time > 0


if detect_jax():
    def test_jax_timer():
        t_timer_engine('jax')

if detect_cuda():
    def test_cupy_timer():
        t_timer_engine('cupy')
