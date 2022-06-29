import os

from pathlib import Path

import click
import numpy as np

from scipy.sparse import linalg as sparse_linalg, issparse

from xrun.data.loader import load_dataset


def compute_squared_frobenius_norm(file_path: Path) -> float:
    X = load_dataset(str(file_path))
    if issparse(X):
        frob_norm = sparse_linalg.norm(X, ord="fro")
    else:
        frob_norm = np.linalg.norm(X, ord="fro")
    squared_frob_norm = np.square(frob_norm)
    return squared_frob_norm


def calc_nytimes_frob(data_dir: str) -> None:
    data_paths = list(Path(data_dir).glob("docword.nytimes.t*.gz"))
    for data_path in data_paths:
        output_path = f"{data_path}-nytsqrfrob.txt"
        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Skipping...")
        else:
            squared_frob_norm = compute_squared_frobenius_norm(data_path)
            with open(output_path, "w") as fp:
                fp.write(f"{squared_frob_norm}")
    

@click.command(help="Computes the squared Frobenius norm of NYTimes datasets.")
@click.option(
    "-d",
    "--data-dir",
    type=click.STRING,
    required=True,
)
def main(data_dir: str):
    calc_nytimes_frob(data_dir=data_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
