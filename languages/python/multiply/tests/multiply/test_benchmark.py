from multiply.benchmark import benchmark
from multiply.multiply import detect_cuda
import numpy as np


def spin(N):
    sum([i * i for i in range(N)])


def test_benchmark():
    assert (benchmark(spin, 10000) > benchmark(spin, 1000))


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

def test_jax_timer():
    t_timer_engine('jax-cpu')

def test_torch_timer():
    t_timer_engine('torch-cpu')

if detect_cuda():
    def test_cupy_timer():
        t_timer_engine('cupy')

    def test_torch_gpu_timer():
        t_timer_engine('torch-gpu')

    def test_jax_gpu_timer():
        t_timer_engine('jax-gpu')
