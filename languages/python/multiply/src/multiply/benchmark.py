import numpy as np
import time

class Timer:
    def __init__(self, warmup = 10, repeat = 100):
        self.warmup = warmup
        self.repeat = repeat

    def timeit(self, fn, *args):
        for _ in range(self.warmup):
            fn(*args)
        tic = time.perf_counter()
        for _ in range(self.repeat):
            fn(*args)
        toc = time.perf_counter()
        return (toc - tic)/self.repeat
    
def benchmark(fn, *args):
    timer = Timer(warmup=3, repeat=5)
    return timer.timeit(fn, *args)

def benchmark_range(fn, ordinates):
    timer = Timer(warmup=3, repeat=5)
    # measure a function, which takes a variety of matrices as inputs
    times = np.vectorize(lambda x: timer.timeit(fn, x))(ordinates.astype(int))
    return np.vstack([ordinates, times])