import re, os

from pathlib import Path
from pprint import pprint
from typing import Dict, List

import click
import numpy as np

from xrun.data.loader import load_dataset


def compute_squared_frobenius_norm(X, reduced_file_path: Path) -> float:
    X_reduced = load_dataset(str(reduced_file_path))
    frob_norm = np.linalg.norm(X - X_reduced, ord="fro")
    squared_frob_norm = np.square(frob_norm)
    return squared_frob_norm


def compute_and_persist(original_file_path: Path, reduced_file_paths: List[Path]):
    X = load_dataset(str(original_file_path))
    for reduced_file_path in reduced_file_paths:
        print(f"Processing {reduced_file_path}")
        squared_frob_norm = compute_squared_frobenius_norm(X, reduced_file_path)
        print(f"  => Frobenius norm of difference: {squared_frob_norm}")
        output_path = f"{reduced_file_path}-sqrfrob.txt"
        with open(output_path, "w") as fp:
            fp.write(f"{squared_frob_norm}")


def get_original_reduced_mappings(data_dir: Path) -> Dict[str, List[Path]]:
    reduced_data_paths = list(data_dir.glob("*-svd-d*"))
    original_reduced_map: Dict[str, List[Path]] = dict()
    for reduced_file_path in reduced_data_paths:
        if not re.search(r"-svd-d\d+\.txt\.gz$", str(reduced_file_path)):
            continue
        if os.path.exists(f"{reduced_file_path}-sqrfrob.txt"):
            print(f"Already calculated for {reduced_file_path}. Skipping ...")
            continue
        original_file_path = re.sub(r"-svd-d\d+\.txt\.gz", "", str(reduced_file_path))
        if original_file_path not in original_reduced_map:
            original_reduced_map[original_file_path] = []
        original_reduced_map[original_file_path].append(reduced_file_path)
    return original_reduced_map


def calc_frob_diff(data_dir: str) -> None:
    original_reduced_map = get_original_reduced_mappings(Path(data_dir))
    pprint(original_reduced_map)

    for orginal_file_path, reduced_file_paths in original_reduced_map.items():
        compute_and_persist(Path(orginal_file_path), reduced_file_paths)


@click.command(help="Computes the Frobenius norm of difference between the original dataset and PCA transformed dataset.")
@click.option(
    "-d",
    "--data-dir",
    type=click.STRING,
    required=True,
)
def main(data_dir: str):
    calc_frob_diff(data_dir=data_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
