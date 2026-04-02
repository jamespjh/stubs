import argparse
import logging

from multiply.payloads import matrix_at_size
from multiply.payloads import multiply_matrices
from multiply.benchmark import benchmark_engine
from .array_abstraction import detect_cuda, detect_metal, detect_jax

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
    logger.info("Using JAX backend")
    jaxc = matmul(size, 'jax-cpu')
    logger.info("Using JAX GPU backend")
    jaxg = matmul(size, 'jax-gpu')
    logger.info("Using torch CPU backend")
    torch = matmul(size, 'torch-cpu')
    print("Engine: Time/s:")
    print("---------------")
    print(f"Python : {python:.3g}")
    print(f"NumPy  : {numpy:.3g}")
    print(f"Numba  : {numba:.3g}")
    print(f"Torch  : {torch:.3g}")
    print(f"JAX-CPU: {jaxc:.3g}")
    print(f"JAX-GPU: {jaxg:.3g}")
    if detect_cuda():
        logger.info("Using CUDA backend")
        cupy = matmul(size, 'cupy')
        logger.info("Using torch GPU backend")
        torch_gpu = matmul(size, 'torch-gpu')
        print(f"CuPy   : {cupy:.3g}")
        print(f"Trch-GP: {torch_gpu:.3g}")
    else:
        logger.info("CUDA not detected, skipping CuPy and torch-cuda benchmarks")
        print("CuPy   : n/a")
        print("Trch-GP: n/a")
    if detect_metal():
        logger.info("Using Metal backend")
        metal = matmul(size, 'mlx')
        print(f"Metal  : {metal:.3g}")
    else:
        logger.info("Metal not detected, skipping Metal benchmark")
        print("Metal  : n/a")

def matmul(size, engine):
    x = matrix_at_size(size, engine)
    y = matrix_at_size(size, engine)

    return benchmark_engine(multiply_matrices, engine, x, y, engine)
