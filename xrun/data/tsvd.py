from timeit import default_timer as timer
from typing import List

import click
import numpy as np
import pandas as pd

from scipy import linalg
from scipy.sparse import linalg as sparse_linalg, issparse
from sklearn.utils.extmath import svd_flip, safe_sparse_dot

from xrun.data.loader import load_dataset


def perform_projection(X, target_dim: int):
    print(f"Computing SVD with target dimensions {target_dim}...")
    start_time = timer()
    VT = None

    # U: Unitary matrix having left singular vectors as columns. 
    # Sigma: The singular values, sorted in descending order.
    # VT: Unitary matrix having right singular vectors as rows.
    if issparse(X):
        print(" - Using ARPACK to compute SVD because input matrix is sparse.")
        U, Sigma, VT = sparse_linalg.eigen.svds(A=X, k=target_dim, solver='arpack')

        # svds sorts the singular values/vectors in ascending order, reverse it
        Sigma = Sigma[::-1]
        U, VT = svd_flip(U[:, ::-1], VT[::-1])

        # Perform the transformation: X * V
        X_transformed = safe_sparse_dot(X, VT.T)
    else:
        print(" - Using LAPACK to compute SVD because input matrix is dense.")
        U, Sigma, VT = linalg.svd(
            a=X,
            full_matrices=False,
            overwrite_a=True,
            lapack_driver='gesdd'
        )

        # Only take the k singular vectors corresponding to the largest singular values
        # by zeroing out the rest of singular values in `Sigma`.
        Sigma[target_dim:] = 0

        # X_transformed = U * Sigma * V^T
        X_transformed = np.dot(U, np.dot(np.diag(Sigma), VT))

    end_time = timer()
    print(f" - Completed {end_time - start_time:.2f} secs. Transformed shape: {X_transformed.shape}.")

    return X_transformed, VT


def persist_to_disk(data: np.ndarray, output_path: str) -> None:
    print(f"Storing data to {output_path}...")
    start_time = timer()
    df_data = pd.DataFrame(data)
    df_data.to_csv(output_path, index=False, header=False)
    end_time = timer()
    print(f" - Completed in : {end_time - start_time:.2f} secs")


def compute_squared_frobenius_norm(data: np.ndarray) -> float:
    if issparse(data):
        frob_norm = sparse_linalg.norm(data, ord="fro")
    else:
        frob_norm = np.linalg.norm(data, ord="fro")
    squared_frob_norm = np.square(frob_norm)
    return squared_frob_norm


def compute_mass(X: np.ndarray, X_reduced: np.ndarray) -> float:
    if X.shape != X_reduced.shape:
        # Special case for BoW datasets where dimensions mismatch
        # For NYTimes dataset, the PCA transformed dataset has fewer dimensions than
        # the original dataset because storing 500,000 * 102,000 values is not
        # practical. To add back the mass that are projected away, we compute the
        # quantity ||A||^2_F - ||B||^2_F where A is the original data matrix
        # and B is the PCA transformed data matrix.
        X_frob_norm = compute_squared_frobenius_norm(X)
        X_reduced_frob_norm = compute_squared_frobenius_norm(X_reduced)
        return X_frob_norm - X_reduced_frob_norm
    else:
        return compute_squared_frobenius_norm(X - X_reduced)


def reduce_dim(input_path: str, target_dims: List[int]) -> None:
    X = load_dataset(input_path)
    for target_dim in target_dims:
        start_time = timer()
        X_transformed, VT = perform_projection(X, target_dim)
        end_time = timer()
        with open(f"{input_path}-svd-d{target_dim}-running-time.txt", "w") as fp:
            fp.write(f"{end_time - start_time}")
        reduced_dim_file_path = f"{input_path}-svd-d{target_dim}.txt.gz"
        persist_to_disk(X_transformed, reduced_dim_file_path)
        persist_to_disk(VT, f"{input_path}-svd-d{target_dim}-vt.txt.gz")
        mass = compute_mass(X=X, X_reduced=X_transformed)
        with open(f"{reduced_dim_file_path}-sqrfrob.txt", "w") as fp:
            fp.write(f"{mass}")


def validate_target_dims(ctx, param, value):
    if value is None:
        raise Exception("Invalid target dimension")
    ret_val = []
    for s in value.split(","):
        try:
            ret_val.append(int(s))
        except ValueError:
            raise Exception(f"Dimension is not an integer: {s}")
    return ret_val


@click.command(help="Dimensionality Reduction via SVD.")
@click.option(
    "-i",
    "--input-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-d",
    "--target-dims",
    required=True,
    callback=validate_target_dims
)
def main(input_path: str, target_dims: List[int]):
    reduce_dim(
        input_path=input_path,
        target_dims=target_dims,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
