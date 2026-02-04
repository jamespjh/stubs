from multiply.matrix_factory import random_matrix

def test_python_factory():
    mat = random_matrix(10, engine='python')
    assert len(mat) == 10
    assert len(mat[0]) == 10
    for row in mat:
        for val in row:
            assert 0.0 <= val < 1.0

def test_numpy_factory():
    mat = random_matrix(10, engine='numpy')
    assert mat.shape == (10, 10)
    assert(all(mat.flatten() >= 0.0))
    assert(all(mat.flatten() < 1.0))
