# Requires some complex mocking to test properly

# Just testing the engines that work without GPU

def test_payload_python():
    from multiply.payloads import multiply_at_size

    size = 10
    result = multiply_at_size(size, engine='python')
    assert len(result) == size
    assert len(result[0]) == size


def test_payload_numpy():
    from multiply.payloads import multiply_at_size

    size = 10
    result = multiply_at_size(size, engine='numpy')
    assert result.shape == (size, size)
