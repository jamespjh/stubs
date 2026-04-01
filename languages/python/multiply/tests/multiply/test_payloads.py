# Requires some complex mocking to test properly

# Just testing the engines that work without GPU

def t_payload(engine):
    from multiply.payloads import multiply_at_size

    size = 10
    result = multiply_at_size(size, engine=engine)
    if engine in ['python', 'numba']:
        assert isinstance(result, list)
        assert len(result) == size
        assert len(result[0]) == size
    else:
        assert result.shape == (size, size)

def test_payload_python():
    t_payload('python')


def test_payload_numpy():
   t_payload('numpy')


def test_payload_jax():
    t_payload('jax-cpu')

def test_payload_numba():
    t_payload('numba')