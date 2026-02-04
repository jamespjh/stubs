# Example stub code for apple MLX to demonstrate GPU speedup

def matmul(size):
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape = (size, size))
    b = mx.random.uniform(shape = (size, size))
    def fn():
        c = a @ b
        mx.sync()
    return fn