# Scale up and modularise

from random import random
import numpy as np

def random_matrix(size, engine):
    if engine == 'mlx':
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        return mx.random.uniform(shape = (size, size))
    elif engine == 'cupy':
        pass
    elif engine == 'python':
        max_python_size = 512
        if size > max_python_size:
            raise ValueError(f"Size {size} is too large for native-python. Maximum size is {max_python_size}.")
        return [[
            random() 
            for _ in range(size)]
            for _ in range(size)]
    elif engine == 'numpy':
        return np.random.uniform(size = (size, size))
    raise ValueError(f"Unknown engine: {engine}")
    
def stack_matrices(copies, chunk, engine):
    if engine == 'mlx':
        import mlx.core as mx
        return mx.concatenate((mx.concatenate((chunk,)*3, axis=1),)*3, axis=0)
    elif engine == 'numpy':
        return np.hstack(np.vstack([chunk]*copies)*copies)
    raise ValueError(f"Unknown engine: {engine}")

def matrix_at_size(size, engine):
    # Construct in chunks to save time
    max_rand_size = 8192
    max_size = 24576

    if size > max_size:
        raise ValueError(f"Size {size} is too large. Maximum size is {max_size}.")
    if size > max_rand_size:
        #Creating to size 24576, then truncating to requested
        chunk = random_matrix(max_rand_size, engine)
        res = stack_matrices(3, chunk, engine)
        return res[:size, :size]
    else:
        return random_matrix(size, engine)

