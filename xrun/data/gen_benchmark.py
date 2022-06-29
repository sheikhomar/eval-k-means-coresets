import os
from timeit import default_timer as timer

import click
import numpy as np
import pandas as pd

from numba import jit

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def generate_benchmark(k: int, alpha: int, beta: float):
    """Generate benchmark dataset.

    Parameters
    ----------
    k: int
        The size of the initial block which is a square k-by-k matrix.
    alpha: int
        The number of the column blocks.
    beta: float:
        The factor for which to scale the column box.
    Returns
    -------
    np.array
        Returns a matrix of size (k^alpha x alpha*k)
    """
    # Create NxD matrix
    n = k ** alpha
    d = alpha * k
    data = np.zeros((n, d))

    # Construct the first N-by-k left-side of the matrix
    for i in range(n):
        for j in range(k):
            value = 0
            if i % k == j:
                value = (k-1) / k
            else:
                value = -1/k
            data[i,j] = value

    # Fill the rest by using the copy-stack operation
    for j in range(k, d):
        for i in range(n):
            copy_i = i // k
            copy_j = j - k
            data[i,j] = data[copy_i, copy_j]
    
    # Apply columnblock-level scaling factor beta
    if beta > 1:
        for i in range(alpha):
            start_col = i*k
            end_col = i*k + k
            beta_val = np.power(beta, float(-i))
            data[:, start_col:end_col] *= beta_val

    return data


def gen_benchmark(block_size: int, alpha: int, beta: int, output_dir: str):
    print("Generating benchmark dataset...")
    start_time = timer()
    dataset = generate_benchmark(
        k=block_size,
        alpha=alpha,
        beta=beta,
    )
    end_time = timer()
    print(f"Dataset of shape {dataset.shape} (size: {dataset.nbytes / 1024 / 1024:0.0f} MB) generated in {end_time - start_time:.2f} secs")

    print("Storing data on disk...")
    start_time = timer()
    df_data = pd.DataFrame(dataset)
    output_path = os.path.join(output_dir, f"benchmark-k{block_size}-alpha{alpha}-beta{beta:0.2f}.txt.gz")
    df_data.to_csv(output_path, index=False, header=False)
    end_time = timer()
    print(f"Elapsed time: {end_time - start_time:.2f} secs")


@click.command(help="Generates benchmark dataset.")
@click.option(
    "-k",
    "--block-size",
    type=click.INT,
    required=True,
)
@click.option(
    "-a",
    "--alpha",
    type=click.INT,
    required=True,
)
@click.option(
    "-b",
    "--beta",
    type=click.FLOAT,
    required=False,
    default=1.0,
)
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
def main(block_size: int, alpha: int, beta: float, output_dir: str):
    gen_benchmark(
        block_size=block_size,
        alpha=alpha,
        beta=beta,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
