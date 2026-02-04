import argparse
import logging

from multiply.payloads import matrix_at_size
from multiply.payloads import multiply_matrices
from multiply.benchmark import benchmark

logger = logging.getLogger(__name__)


def entry():
    parser = argparse.ArgumentParser(
        description='Demonstrate GPU speedup')
    parser.add_argument("--loglevel", default='ERROR',
                        help='Set the logging level' +
                        '(DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument("--size", type=int, default=256,
                        help='Matrix size to use for benchmark' +
                        '(default: 256)')
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)
    multiply_results(args.size)


def multiply_results(size):
    logger.info("Using python backend")
    python = matmul(size, 'python')
    logger.info("Using numpy backend")
    numpy = matmul(size, 'numpy')
    logger.info("Using numba backend")
    numba = matmul(size, 'numba')
    print("Engine: Time/s:")
    print("---------------")
    print(f"Python: {python:.3g}")
    print(f"NumPy : {numpy:.3g}")
    print(f"Numba : {numba:.3g}")
    if detect_cuda():
        logger.info("Using CUDA backend")
        cupy = matmul(size, 'cupy')
        print(f"CuPy  : {cupy:.3g}")
    else:
        logger.info("CUDA not detected, skipping CuPy benchmark")
        print("CuPy  : n/a")
    if detect_metal():
        logger.info("Using Metal backend")
        metal = matmul(size, 'mlx')
        print(f"Metal : {metal:.3g}")
    else:
        logger.info("Metal not detected, skipping Metal benchmark")
        print("Metal : n/a")


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


def matmul(size, engine):
    x = matrix_at_size(size, engine)
    y = matrix_at_size(size, engine)

    return benchmark(multiply_matrices, x, y, engine)
