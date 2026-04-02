
from multiply.multiply import detect_cuda, detect_metal

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

def test_payload_torch():
    t_payload('torch-cpu')

if detect_cuda():
    def test_payload_torch_gpu():
        t_payload('torch-gpu')

    def test_payload_cupy():
        t_payload('cupy')

    def test_payload_jax_gpu():
        t_payload('jax-gpu')

def test_payload_jax():
    t_payload('jax-cpu')

def test_payload_numba():
    t_payload('numba')