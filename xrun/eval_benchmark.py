import os, subprocess

from typing import List, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import re
import pandas as pd
import numba

import click

from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from scipy.sparse import issparse
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances_chunked
from sklearn.metrics.pairwise import _argmin_min_reduce

from xrun.gen import generate_random_seed
from xrun.data.loader import load_dataset
from xrun.data.run_info import RunInfo
from xrun.data.gen_benchmark import generate_benchmark

BENCHMARK_FILE_NAME = "benchmark-distortion.txt"


def unzip_file(input_path: Path) -> Path:
    output_path = Path(os.path.splitext(input_path)[0])
    if not output_path.exists():
        print(f"Unzipping file {input_path}...")
        p = subprocess.Popen(
            args=["gunzip", "-k", str(input_path)],
            start_new_session=True
        )
        p.wait()
    assert(output_path.exists())
    return output_path


datasets = dict()
def load_data(run_info: RunInfo):
    dataset_path = run_info.dataset_path
    if dataset_path not in datasets:
        dataset = load_dataset(dataset_path)
        datasets[dataset_path] = dataset
    return datasets[dataset_path]


def load_run_info(experiment_dir: Path) -> Optional[RunInfo]:
    run_file_paths = list(experiment_dir.glob("*.json"))
    if len(run_file_paths) != 1:
        print(f"Expected a single run file in {experiment_dir} but found {len(run_file_paths)} files.")
        return None
    return RunInfo.load_json(run_file_paths[0])


def find_unprocesses_result_files(results_dir: str) -> List[Path]:
    search_dir = Path(results_dir)
    output_paths = list(search_dir.glob('**/results.txt.gz'))
    return_paths = []
    for file_path in output_paths:
        run_info = load_run_info(file_path.parent)
        already_evaluated = os.path.exists(file_path.parent / BENCHMARK_FILE_NAME)
        if not already_evaluated and run_info is not None:
            return_paths.append(file_path)
    return return_paths

@numba.njit
def get_cluster_member_indices(k: int, alpha: int, block_index: int, cluster_label: int):
    """Computes the members of cluster `i` induces by block `a`.
    
    Computes C_i^a where `a` is the block index and `i` is the cluster label.
    """
    n = k ** alpha
    a = block_index

    cluster_labels = np.array([(n // k**a) % k  for n in range(n)])
    indices = np.where(cluster_labels == cluster_label)[0]
    return indices

def pairwise_distances_argmin_min_fast(X, Y, metric):
    indices, values = zip(*pairwise_distances_chunked(
        X, Y, reduce_func=_argmin_min_reduce, metric=metric, working_memory=10*1024))
    indices = np.concatenate(indices)
    values = np.concatenate(values)
    return indices, values


def compute_benchmark_costs(run_info: RunInfo, coreset_path: Path) -> None:
    experiment_dir = coreset_path.parent
    coreset_cost_path = experiment_dir / "coreset_cost.txt"
    real_cost_path = experiment_dir / "real_cost.txt"
    benchmark_distortion_path = experiment_dir / BENCHMARK_FILE_NAME

    if benchmark_distortion_path.exists():
        print("Costs are already computed! Existing...")
        return

    parsed_str = re.findall(r"benchmark-k(\d+)-alpha(\d+)-beta(\d+.\d+)", run_info.dataset_path)[0]
    k = int(parsed_str[0])
    alpha = int(parsed_str[1])
    beta = float(parsed_str[2])

    # print(f"Loading benchmark dataset from {run_info.dataset_path}")
    # benchmark = load_dataset(run_info.dataset_path)
    print(f"Generating benchmark dataset with k={k}, alpha={alpha}, beta={beta:0.1f}")
    benchmark = generate_benchmark(k=k, alpha=alpha, beta=beta)

    print(f"Loading coreset from {coreset_path}...")
    computed_coreset = np.loadtxt(
        fname=coreset_path,
        dtype=np.double,
        delimiter=" ",
        skiprows=1,
        unpack=False
    )
    coreset_weights = computed_coreset[:,0]
    coreset_points = computed_coreset[:,1:]

    print("Snapping coreset points to their closest benchmark points")
    start_time = datetime.now()
    # Since it is not always the case that set of coreset points Ω consist only of input points,
    # we need to snap the computed coreset points to their closest benchmark points. The snapped
    # indices referer to the points in the benchmark dataset.
    snapped_indices, _ = pairwise_distances_argmin_min_fast(X=coreset_points, Y=benchmark, metric="euclidean")
    end_time = datetime.now()
    print(f" - Finished in {end_time - start_time}")

    df_costs = pd.DataFrame(columns=[
        "block_index", "epsilon", "n_non_deficient_centers",
        "coreset_cost", "real_cost", "distortion"
    ], dtype=float)

    for a in range(alpha):
        for epsilon in [0.01, 0.05, 0.1, 0.3, 0.5]:
            start_time = datetime.now()

            # Compute non-deficient centers for block index `a` and epsilon
            non_deficient_cluster_means = []

            for i in range(k):
                # Compute members of the cluster `i` induced by block `a`: C_i^a
                cluster_members = get_cluster_member_indices(k=k, alpha=alpha, block_index=a, cluster_label=i)

                # Compute the size of the cluster: |C_i^a|
                n_cluster_members = cluster_members.shape[0]

                # Compute the intersection between cluster members and the snapped coreset points: C_i^a ∩ Ω.
                # We want to find points in the compute coreset that belong to the cluster `i` induced by block `a`.
                snapped_indices_in_cluster = np.intersect1d(snapped_indices, cluster_members)

                # Find indices in the compute coreset for all points in C_i^a ∩ Ω.
                coreset_indices_in_cluster = np.where(np.isin(snapped_indices, snapped_indices_in_cluster))[0]

                # Compute the mass of points of C_i^a in Ω
                mass = np.sum(coreset_weights[coreset_indices_in_cluster])

                # print(f"Computed mass for a={a}, i={i+1} is {mass}")

                # Compute |C_i^a|(1-ϵ)
                non_deficient_threshold = n_cluster_members * (1 - epsilon)

                # print(f" For epsilon {epsilon}, the non-deficient threshold is {non_deficient_threshold}")

                if mass >= non_deficient_threshold:
                    # print(" - Adding cluster mean to the solution")
                    cluster_mean = np.mean(benchmark[cluster_members], axis=0)
                    non_deficient_cluster_means.append(cluster_mean)
            
            # Let the non-deficient cluster means be the centers
            center_points = np.array(non_deficient_cluster_means)

            # If only deficient clusters are found then record zero costs and distortion.
            if center_points.shape[0] == 0:
                df_costs = df_costs.append({
                    "block_index": a,
                    "epsilon": epsilon,
                    "n_non_deficient_centers": center_points.shape[0],
                    "coreset_cost": 0,
                    "real_cost": 0, 
                    "distortion": 0,
                }, ignore_index=True)
            else:
                # Compute coreset cost based on the coreset points and the non-deficient cluster means
                _, closest_sqdist_coreset_centers = pairwise_distances_argmin_min_fast(X=coreset_points, Y=center_points, metric="sqeuclidean")

                # The coreset cost is the sum of weighted squared distances to the closest centers.
                coreset_cost = np.sum(coreset_weights * closest_sqdist_coreset_centers)

                # Compute the real cost based on the benchmark points and the non-deficient cluster means
                _, closest_sqdist_benchmark_centers = pairwise_distances_argmin_min_fast(X=benchmark, Y=center_points, metric="sqeuclidean")

                # The real cost is the sum of squared distances to the closest centers.
                real_cost = np.sum(closest_sqdist_benchmark_centers)

                distortion = max(real_cost / coreset_cost, coreset_cost / real_cost)

                end_time = datetime.now()
                print(f"For a={a} epsilon={epsilon:0.2f}, the distortion is: {distortion:0.2f} - Finished in {end_time - start_time}")

                df_costs = df_costs.append({
                    "block_index": a,
                    "epsilon": epsilon,
                    "n_non_deficient_centers": center_points.shape[0],
                    "coreset_cost": coreset_cost,
                    "real_cost": real_cost, 
                    "distortion": distortion,
                }, ignore_index=True)

    df_costs.to_csv(str(experiment_dir / "benchmark-costs.csv"), index=False)

    # Find the row with the largest distortion.
    worst = df_costs.loc[df_costs.distortion.argmax()]
    with open(coreset_cost_path, "w") as f:
        f.write(str(worst.coreset_cost))
    with open(real_cost_path, "w") as f:
        f.write(str(worst.real_cost))
    with open(benchmark_distortion_path, "w") as f:
        f.write(str(worst.distortion))

def process_result(index: int, n_total: int, result_path: Path):
    print(f"Processing file {index+1} of {n_total}: {result_path}")
    experiment_dir = result_path.parent

    run_info = load_run_info(experiment_dir)
    if run_info is None:
        print("Cannot process results file because run file is missing.")
        return

    if not os.path.exists(run_info.dataset_path):
        print(f"Dataset path: {run_info.dataset_path} cannot be found. Skipping...")
        return

    compute_benchmark_costs(run_info=run_info, coreset_path=result_path)    


def eval_benchmark(results_dir: str, n_jobs: int) -> None:
    output_paths = find_unprocesses_result_files(results_dir)
    total_files = len(output_paths)
    Parallel(n_jobs=n_jobs)(
        delayed(process_result)(index=index, n_total=total_files, result_path=result_path)
        for index, result_path in enumerate(output_paths)
    )


@click.command(help="Evaluate algorithm on benchmark dataset.")
@click.option(
    "-r",
    "--results-dir",
    type=click.STRING,
    required=True,
)
@click.option(
    "-j",
    "--n-jobs",
    type=click.INT,
    required=False,
    default=1
)
def main(results_dir: str, n_jobs: int) -> None:
    eval_benchmark(results_dir=results_dir, n_jobs=n_jobs)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
