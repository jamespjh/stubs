from multiply.matrix_factory import matrix_at_size

def python_multiply(x, y):
   
    return [[
        sum([a * b for a, b in zip(row_x, col_y)])
        for col_y in zip(*y)] 
        for row_x in x]

import numba
@numba.njit
def numba_python_multiply(x, y):
    yt = [[
        y[j][i] for j in range(len(y))]
        for i in range(len(y[0]))] #zip(*y) will not numba jit
    return [[
        sum([a * b for a, b in zip(row_x, col_y)])
        for col_y in yt] 
        for row_x in x]

def multiply_matrices(x, y, engine):
    if engine == 'python':
        return python_multiply(x, y)
    if engine == 'numba':
        return numba_python_multiply(x, y)
    if engine not in ['mlx', 'numpy']:
        raise ValueError(f"Unknown engine: {engine}")
    res = x @ y
    if engine == 'mlx':
        import mlx.core as mx
        return mx.eval(res)
    else:
        return res


def multiply_at_size(size, engine):
    x = matrix_at_size(size, engine)
    y = matrix_at_size(size, engine)
    return multiply_matrices(x, y, engine)
