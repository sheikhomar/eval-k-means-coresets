import os

from timeit import default_timer as timer
from typing import List, Optional

import numpy as np

from pathlib import Path

import click

from sklearn.metrics import pairwise_distances_argmin_min

from xrun.data.loader import load_dataset
from xrun.data.run_info import RunInfo


CORESET_CENTER_FILE_NAME = "coreset_center_cost.txt"


def compute_coreset_center_cost(original_points: np.ndarray, coreset_file_path: Path) -> Path:
    cost_file_path = coreset_file_path.parent / CORESET_CENTER_FILE_NAME
    if cost_file_path.exists():
        return cost_file_path

    print(f"Loading coreset from {coreset_file_path}... ", end='')
    start_time = timer()
    coreset = np.loadtxt(fname=coreset_file_path, dtype=np.double, delimiter=' ', skiprows=1)
    coreset_weights = coreset[:,0]
    coreset_points = coreset[:,1:]
    print(f" done in {timer() - start_time} seconds.")

    print("Computing coreset center cost... ")
    start_time = timer()
    # Compute squared Euclidean distances between all original points and coreset points
    closest_coreset_points, distances = pairwise_distances_argmin_min(
        X=original_points, 
        Y=coreset_points,
        metric="sqeuclidean"
    )

    # Sum up all the distances
    cost = np.sum(distances)
    print(f" - Computed cost = {cost} in {timer() - start_time} seconds.")
    
    with open(cost_file_path, "w") as f:
        f.write(str(cost))
    return cost_file_path


def find_unprocesses_run_files(input_dir: str) -> List[Path]:
    run_info_paths = list(sorted(Path(input_dir).glob('**/*.json')))
    return_paths = []
    for run_info_path in run_info_paths:
        costs_computed = os.path.exists(run_info_path.parent / CORESET_CENTER_FILE_NAME)
        already_included = len(list(filter(lambda p: p.parent.parent == run_info_path.parent.parent, return_paths))) > 0
        if not costs_computed and not already_included:
            return_paths.append(run_info_path)
    return return_paths


@click.command(help="Compute costs when coresets are used as centers.")
@click.option(
    "-r",
    "--results-dir",
    type=click.STRING,
    required=True,
)
def main(results_dir: str) -> None:
    dataset_cache = dict()
    run_info_paths = find_unprocesses_run_files(results_dir)
    total_files = len(run_info_paths)
    for index, run_info_path in enumerate(run_info_paths):
        coreset_path = run_info_path.parent / "results.txt.gz"
        run_info = RunInfo.load_json(run_info_path)

        print(f"Processing file {index+1} of {total_files}: {coreset_path}")
        
        if not coreset_path.exists():
            print(f"Cannot process results file because coreset file is missing: {run_info_path}")
            continue

        if not os.path.exists(run_info.dataset_path):
            print(f"Dataset path: {run_info.dataset_path} cannot be found. Skipping...")
            continue

        dataset_path = run_info.dataset_path
        if dataset_path not in dataset_cache:
            dataset_cache[dataset_path] = load_dataset(dataset_path)
        original_dataset = dataset_cache[dataset_path]

        compute_coreset_center_cost(
            original_points=original_dataset,
            coreset_file_path=coreset_path,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
