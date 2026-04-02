import numpy as np
import jax
import numba
from random import random
from operator import mul
from functools import reduce

jax_engines = ['jax-cpu', 'jax-gpu', 'jax-metal']
mlx_engines = ['mlx-cpu', 'mlx-gpu']
torch_engines = ['torch-cpu', 'torch-gpu']
valid_engines = [
    'python',
    'numba',
    'numpy',
    'cupy'] + jax_engines + mlx_engines + torch_engines

def random_python_matrix(size):
    if len(size) != 2:
        raise ValueError("Python engine only supports 2D matrices.")
    return [[
        random()
        for _ in range(size[0])]
        for _ in range(size[1])]

@numba.njit
def numba_python_matrix(size):
    if len(size) != 2:
        raise ValueError("Python engine only supports 2D matrices.")
    return numba.typed.List([
        numba.typed.List([
            random()
            for _ in range(size[0])])
        for _ in range(size[1])])


class ArrayAbstraction:

    def __init__(self, engine):
        self.engine = engine
        if engine == 'python':
            self.np = None
            self.random = random
        elif engine == 'numba':
            self.np = None
            self.random = random
        elif engine == 'numpy':
            self.np = np
            self.random = np.random
        elif engine == 'cupy':
            import cupy as cp
            self.np = cp
            self.random = cp.random
        elif engine in torch_engines:
            import torch
            self.np = torch
            self.np.array = torch.as_tensor
            self.random = torch.rand
        elif engine in jax_engines:
            import jax.numpy as jnp
            import jax.random as jrandom
            self.np = jnp
            self.random = jrandom
            self.key = jrandom.key(0)
            if engine == 'jax-metal':
                self.jax_device = jax.devices("METAL")[0]
            elif engine == 'jax-gpu':
                self.jax_device = jax.devices("gpu")[0]
            else:
                self.jax_device = jax.devices("cpu")[0]
        elif engine in mlx_engines:
            import mlx.core as mx
            if engine == 'mlx-cpu':
                mx.set_default_device(mx.cpu)
            else:
                mx.set_default_device(mx.gpu)
            self.np = mx
            self.random = mx.random
        else:
            raise ValueError(
                f"Unknown engine '{engine}'. Valid engines "
                f"are: {valid_engines}.")

    def array(self, data):
        """Create an array in the appropriate engine."""
        res = self.np.array(data)
        if self.engine in ['jax-cpu', 'jax-gpu', 'jax-metal']:
            res = jax.device_put(res, self.jax_device)
        if self.engine == 'torch-gpu':
            res = res.to('cuda')
        return res

    def random_array(self, shape, min=0.0, max=1.0):
        """Generate a random array of the given shape."""
        if self.engine == 'python':
            max_python_size = 1024
            if reduce(mul, shape) > max_python_size*max_python_size:
                raise ValueError(
                    f"Size {shape} is too large for native-python." +
                    f"Maximum size is {max_python_size}.")
            return random_python_matrix(shape)
        elif self.engine == 'numba':
            return numba_python_matrix(shape)
        elif self.engine in ['numpy', 'cupy']:
            return self.random.uniform(low=min, high=max, size=shape)
        elif self.engine in jax_engines:
            self.key, subkey = self.random.split(self.key)
            res = self.random.uniform(subkey, shape,
                                      minval=min,
                                      maxval=max)
            res = jax.device_put(res, self.jax_device)
            return res
        elif self.engine in mlx_engines:
            res = self.random.uniform(shape=shape, low=min, high=max)
            return res
        elif self.engine in torch_engines:
            res = self.random(size=shape) * (max - min) + min
            if self.engine == 'torch-gpu':
                res = res.to('cuda')
            return res
        else:
            raise ValueError(
                f"Unknown engine '{self.engine}'."
                f"Valid engines: {valid_engines}.")


def detect_cuda():
    try:
        import cupy
    except ImportError:
        return False
    return cupy.cuda.is_available()


def detect_metal():
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return mx.metal.is_available()


def detect_jax():
    try:
        import jax
    except ImportError:
        return False
    return True


def detect_torch_cuda():
    import torch
    return torch.cuda.is_available()


def clear_gpu_cache():
    if detect_cuda():
        import cupy
        cupy.cuda.Device().synchronize()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    if detect_jax():
        import jax
        [buf.delete() for buf in jax.live_arrays()]
    if detect_torch_cuda():
        import torch
        torch.cuda.empty_cache()