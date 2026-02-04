import mlx.core as mx
from multiply.matrix_factory import matrix_at_size

def multiply_matrices(x, y, engine='mlx'):
    if engine == 'python':
        return [[
            sum(a * b for a, b in zip(row_x, col_y)) 
            for col_y in zip(*y)] 
            for row_x in x]
    res = x @ y
    if engine == 'mlx':
        return mx.eval(res)
    else:
        return res
    
def multiply_at_size(size, engine='mlx'):
    x = matrix_at_size(size, engine)
    y = matrix_at_size(size, engine)
    return multiply_matrices(x, y, engine)