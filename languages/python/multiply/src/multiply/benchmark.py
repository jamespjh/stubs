import numpy as np
import time


class Timer:
    def __init__(self, warmup=10, repeat=100):
        self.warmup = warmup
        self.repeat = repeat

    def timeit(self, fn, *args):
        for _ in range(self.warmup):
            fn(*args)
        tic = time.perf_counter()
        for _ in range(self.repeat):
            fn(*args)
        toc = time.perf_counter()
        return (toc - tic) / self.repeat

    def timeit_cu(self, fn, *args):
        from cupyx.profiler import benchmark
        res = benchmark(
            fn,
            n_repeat=self.repeat,
            n_warmup=self.warmup,
            args=args)
        cpu = res.cpu_times.mean()
        gpu = res.gpu_times.mean()
        # Return the maximum of CPU and GPU time as the benchmark result
        return max(cpu, gpu)

    def timeit_engine(self, fn, *args, engine=None):
        if engine in ['cupy', 'jax']:
            return self.timeit_cu(fn, *args)
        else:
            return self.timeit(fn, *args)


def benchmark(fn, *args):
    timer = Timer(warmup=3, repeat=5)
    return timer.timeit(fn, *args)


def cupy_benchmark(fn, *args):
    from cupyx.profiler import benchmark
    res = benchmark(fn, n_repeat=5, n_warmup=3, args=args)
    cpu = res.cpu_times.mean()
    gpu = res.gpu_times.mean()
    # Return the maximum of CPU and GPU time as the benchmark result
    return max(cpu, gpu)


def benchmark_range(fn, ordinates, engine=None):
    timer = Timer(warmup=3, repeat=5)
    # measure a function, which takes a variety of matrices as inputs
    times = np.vectorize(
        lambda x: timer.timeit_engine(
            fn, x, engine=engine))(
        ordinates.astype(int))
    return np.vstack([ordinates, times])
