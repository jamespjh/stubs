# Scale up and modularise

from .array_abstraction import ArrayAbstraction

max_rand_size = 8192
max_size = 24576


def random_matrix(size, engine):
    return ArrayAbstraction(engine).random_array(
        (size, size), min=0.0, max=1.0)


def stack_matrices(copies, chunk, engine):
    np = ArrayAbstraction(engine).np
    return np.concatenate((np.concatenate((chunk,) * 3, axis=1),) * 3, axis=0)


def matrix_at_size(size, engine):
    # Construct in chunks to save time
    if size > max_size:
        raise ValueError(
            f"Size {size} is too large. Maximum size is {max_size}.")
    if size > max_rand_size:
        # Creating to size 24576, then truncating to requested
        chunk = random_matrix(max_rand_size, engine)
        res = stack_matrices(3, chunk, engine)
        return res[:size, :size]
    else:
        return random_matrix(size, engine)
