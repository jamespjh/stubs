from pytest import raises
from multiply.matrix_factory import random_matrix, stack_matrices
from multiply.multiply import detect_jax, detect_cuda, detect_metal

def test_python_factory():
    mat = random_matrix(10, engine='python')
    assert len(mat) == 10
    assert len(mat[0]) == 10
    for row in mat:
        for val in row:
            assert 0.0 <= val < 1.0


def t_factory(engine):
    mat = random_matrix(10, engine)
    assert mat.shape == (10, 10)
    assert (all(mat.flatten() >= 0.0))
    assert (all(mat.flatten() < 1.0))

def t_stack_matrices(engine):
    from multiply.matrix_factory import stack_matrices
    chunk = random_matrix(10, engine='numpy')
    stacked = stack_matrices(3, chunk, engine='numpy')
    assert stacked.shape == (30, 30)

def test_numpy_factory():
    t_factory('numpy')

def test_invalid_engine():
    from multiply.matrix_factory import random_matrix
    with raises(ValueError):
        random_matrix(10, engine='invalid_engine')


if detect_jax():
    def test_jax_factory():
        t_factory('jax')

    def test_stack_jax_matrices():
        t_stack_matrices('jax')

if detect_cuda():
    def test_cupy_factory():
        t_factory('cupy')
    def test_stack_cupy_matrices():
        t_stack_matrices('cupy')

if detect_metal():
    def test_metal_factory():
        t_factory('mlx')
    def test_stack_metal_matrices():
        t_stack_matrices('mlx')



