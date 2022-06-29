import os, subprocess, shutil, re

from timeit import default_timer as timer
from typing import List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import click
import numba

from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from scipy.sparse import linalg as sparse_linalg, issparse
from sklearn.utils import shuffle
from sklearn.utils.extmath import safe_sparse_dot

from xrun.gen import generate_random_seed
from xrun.data.loader import load_dataset
from xrun.data.run_info import RunInfo
from xrun.eval_benchmark import compute_benchmark_costs

KMEANS_PATH = "kmeans/bin/kmeans.exe"


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


def compute_centers_via_external_kmeanspp(result_file_path: Path) -> Path:
    center_path = result_file_path.parent / "centers.txt"
    
    if center_path.exists() and center_path.stat().st_size > 0:
        return center_path

    if not os.path.exists(KMEANS_PATH):
        raise Exception(f"Program '{KMEANS_PATH}' cannot be found. You can build it: make -C kmeans")

    start_time = timer()

    with open(result_file_path, 'r') as f:
        line1 = next(f)  # Skip the first line
        line2 = next(f)  # Read the first point data to figure out dimensions

    # When counting the number of dimensions for points skip the
    # first entry as it is the weight of the point.
    d = len(line2.split(" ")) - 1  
    k = int(re.findall(r'-k(\d+)-', str(result_file_path))[0])
    random_seed = generate_random_seed()
    command = [
        KMEANS_PATH,
        str(result_file_path),
        str(k),
        str(d),
        str(center_path),
        "0",
        str(random_seed),
    ]
    proc = subprocess.Popen(
        args=command,
        start_new_session=True
    )
    proc.wait()
    end_time = timer()
    print(f"k-means++ centers computed in {end_time - start_time:.2f} secs")
    return center_path


def get_centers(result_file_path: Path) -> np.ndarray:
    for i in range(10):
        centers_file_path = compute_centers_via_external_kmeanspp(result_file_path)
        centers = np.loadtxt(fname=centers_file_path, dtype=np.double, delimiter=' ', skiprows=0)
        center_weights = centers[:,0] 
        center_points = centers[:,1:]

        if np.any(np.isnan(center_points)):
            print("Detected NaN values in the computed centers.")

            center_nan_count = np.count_nonzero(np.isnan(center_points))
            center_inf_count = np.count_nonzero(np.isinf(center_points))

            print(f"- NaN Count: {center_nan_count}")
            print(f"- Inf Count: {center_inf_count}")
            print(f"- NaN: {np.argwhere(np.isnan(center_points))}")
            
            print(f"Removing {centers_file_path}...")
            os.remove(centers_file_path)
        else:
            return center_points

    raise Exception(f"Failed to find centers without NaN values. Giving up after {i+1} iterations!")


datasets = dict()

def load_original_data(run_info: RunInfo):
    dataset_name = run_info.dataset

    if "hardinstance" in dataset_name:
        dataset_name = f"{dataset_name}-k{run_info.k}"
    elif run_info.is_low_dimensional_dataset:
        dataset_name = run_info.original_dataset_name

    if dataset_name not in datasets:
        if run_info.is_low_dimensional_dataset:
            dataset_paths = {
                "census": "data/input/USCensus1990.data.txt",
                "covertype": "data/input/covtype.data.gz",
                "enron": "data/input/docword.enron.txt.gz",
                "nytimes": "data/input/docword.nytimes.txt.gz",
                "caltech101": "data/input/caltech101-sift.txt.gz",
            }
            dataset_path = dataset_paths[dataset_name]
        else:
            dataset_path = run_info.dataset_path
        dataset = load_dataset(dataset_path)
        if issparse(dataset):
            dataset = dataset.todense()
        datasets[dataset_name] = dataset

    return datasets[dataset_name]


def compute_real_cost(data_points: np.ndarray, center_points: np.ndarray, cost_file_path: Path) -> Path:
    if cost_file_path.exists():
        return cost_file_path

    print("Computing real cost... ", end="")

    D = pairwise_distances(data_points, center_points, metric="euclidean", n_jobs=-1)
    D = np.square(D)

    # For each point (w, p) in S, find the distance to its closest center
    dist_closest_centers = np.min(D, axis=1)

    # Weigh the distances and sum it all up
    cost = np.sum(dist_closest_centers)

    print(f"Computed real cost: {cost}")
    
    with open(cost_file_path, "w") as f:
        f.write(str(cost))
    return cost_file_path


def compute_coreset_costs(coreset: np.ndarray, center_points: np.ndarray, added_cost: float, cost_file_path: Path) -> Path:
    if cost_file_path.exists():
        return cost_file_path

    print("Computing coreset cost... ", end='')
    coreset_weights = coreset[:,0]
    coreset_points = coreset[:,1:]

    # Distances between all corset points and center points
    D = pairwise_distances(coreset_points, center_points, metric="sqeuclidean")

    # For each point (w, p) in S, find the distance to its closest center
    dist_closest_centers = np.min(D, axis=1)

    # Weigh the distances and sum it all up
    weighted_cost = np.sum(coreset_weights * dist_closest_centers)

    print(f"Computed coreset cost: {weighted_cost}")

    if added_cost > 0:
        weighted_cost += added_cost
        print(f" Adding additional cost: {added_cost}")
    
    with open(cost_file_path, "w") as f:
        f.write(str(weighted_cost))
    return cost_file_path


def load_run_info(experiment_dir: Path) -> Optional[RunInfo]:
    run_file_paths = list(experiment_dir.glob("*.json"))
    if len(run_file_paths) != 1:
        # print(f"Expected a single run file in {experiment_dir} but found {len(run_file_paths)} files.")
        return None
    return RunInfo.load_json(run_file_paths[0])


def find_unprocesses_result_files(results_dir: str) -> List[Path]:
    search_dir = Path(results_dir)
    output_paths = list(search_dir.glob('**/results.txt.gz'))
    return_paths = []
    for file_path in output_paths:
        costs_computed = np.all([
            os.path.exists(file_path.parent / cfn)
            for cfn in [
                "real_cost.txt", "coreset_cost.txt",
                "real_cost_synthetic.txt", "coreset_cost_synthetic.txt",
                "real_cost_convexsynthetic.txt", "coreset_cost_convexsynthetic.txt"
            ]
        ])
        run_info = load_run_info(file_path.parent)
        if not costs_computed and run_info is not None:
            return_paths.append(file_path)
    return return_paths


def get_added_cost_for_low_dimensional_dataset(run_info: RunInfo) -> float:
    if run_info.is_low_dimensional_dataset:
        if run_info.dataset == "nytimespcalowd":
            # For NYTimes dataset, the PCA transformed dataset has fewer dimensions than
            # the original dataset because storing 500,000 * 102,000 values is not
            # practical. To add back the mass that are projected away, we compute the
            # quantity ||A||^2_F - ||B||^2_F where A is the original data matrix
            # and B is the PCA transformed data matrix.
            sqrfrob_original_path = "data/input/docword.nytimes.txt.gz-nytsqrfrob.txt"
            if not os.path.exists(sqrfrob_original_path):
                raise Exception(f"File not found: {sqrfrob_original_path} Run `python -m xrun.data.nytimes_calc_frob -d data/input` to generate it.")

            sqrfrob_reduced_path = f"{run_info.dataset_path}-nytsqrfrob.txt"
            if not os.path.exists(sqrfrob_reduced_path):
                raise Exception(f"File not found: {sqrfrob_reduced_path} Run `python -m xrun.data.nytimes_calc_frob -d data/input` to generate it.")

            with open(sqrfrob_original_path, "r") as fp:
                sqrfrob_original = float(fp.read())
            with open(sqrfrob_reduced_path, "r") as fp:
                sqrfrob_reduced = float(fp.read())
            return sqrfrob_original - sqrfrob_reduced
        else:
            sqrfrob_file_path = f"{run_info.dataset_path}-sqrfrob.txt"
            if not os.path.exists(sqrfrob_file_path):
                raise Exception(f"File not found: {sqrfrob_file_path} Run `python -m xrun.data.calc_frob_diff -d data/input` to generate extra costs.")
            with open(sqrfrob_file_path, "r") as fp:
                return float(fp.read())
    return 0.0


def expand_dimensionality_nytimes(k: int, centers: np.ndarray):
    start_time = timer()
    VT_path = f"data/input/docword.nytimes.txt.gz-svd-d{k}-vt.txt.gz"
    print(f" - Loading {VT_path}...", end="")
    VT = np.loadtxt(fname=VT_path, dtype=np.double, delimiter=',')
    end_time = timer()
    print(f" Loaded in {end_time - start_time:.2f} secs")

    start_time = timer()
    print(f" - Computing C * V ", end="")
    centers = safe_sparse_dot(centers, VT)
    end_time = timer()
    print(f". Done in {end_time - start_time:.2f} secs")
    return centers


def load_cost_from_file(file_path: Path):
    with open(file_path, "r") as f:
        return float(f.read())


def collect_distortions_of_solutions(costs_dir: Path, n_candidate_solutions: int, file_postfix: str) -> pd.DataFrame:
    costs = []
    for solution_index in range(1, n_candidate_solutions+1):
        real_cost = load_cost_from_file(costs_dir / f"solution{solution_index}-real_cost{file_postfix}.txt")
        coreset_cost = load_cost_from_file(costs_dir / f"solution{solution_index}-coreset_cost{file_postfix}.txt")
        solution_path = costs_dir / f"solution{solution_index}-centers{file_postfix}.txt"
        distortion = max(float(real_cost/coreset_cost), float(coreset_cost/real_cost))
        costs.append({
            "solution_index": solution_index,
            "real_cost": real_cost,
            "coreset_cost": coreset_cost,
            "distortion": distortion,
            "solution_path": solution_path,
        })
    return pd.DataFrame(costs)


def compute_minimum_enclosing_ball(data_matrix: np.ndarray, n_iter: int=100):
    # Implements algorithm from http://cm.bell-labs.co/who/clarkson/coresets2.pdf

    n_points = data_matrix.shape[0]

    # Randomly pick an initial point.
    initial_point_index = np.random.choice(n_points)
    explored_point_indices = [initial_point_index]
    
    # Set the initial point as the center
    center_point = data_matrix[[initial_point_index], :].copy()

    for i in range(1, n_iter+1):
        # Find the point farthest away from current center point
        farthest_point_index = np.argmax(pairwise_distances(center_point, data_matrix))
        explored_point_indices.append(farthest_point_index)
        farthest_point = data_matrix[farthest_point_index, :]

        # Move the center towards the farthest point
        center_point = center_point + (farthest_point - center_point)/(i+1)

    # Compute the radius
    explored_point_indices = list(set(explored_point_indices))
    explored_points = data_matrix[explored_point_indices, :]
    radius = np.max(pairwise_distances(center_point, explored_points))

    return center_point, radius


def generate_arbitrary_point_within_minimum_enclosing_ball(center_point: np.ndarray, radius: float) -> np.ndarray:
    n_dim = center_point.shape[1]

    # Generate a random vector
    random_vector = np.random.multivariate_normal(
        mean=center_point.reshape(-1), 
        cov=np.eye(n_dim)
    )

    # Generate a random length within radius
    length = np.random.uniform(low=1e-5, high=radius)

    random_vector = length * (random_vector / np.linalg.norm(random_vector))

    return random_vector


def generate_random_points_within_minimum_enclosing_ball(data_matrix: np.ndarray, k: int) -> np.ndarray:
    center_point, radius = compute_minimum_enclosing_ball(data_matrix=data_matrix, n_iter=100)

    n_dim = data_matrix.shape[1]
    cluster_centers = np.zeros(shape=(k, n_dim))
    for i in range(k):
        cluster_centers[i] = generate_arbitrary_point_within_minimum_enclosing_ball(
            center_point=center_point,
            radius=radius,
        )

    return cluster_centers


def generate_random_points_within_convex_hull(data_matrix: np.ndarray, k: int, n_samples: int = 2) -> np.ndarray:
    """Generates k random points within the convex hull of the data.

    Note that points are not placed uniformly at random. It is more
    likely to generate points in dense areas of data space.
    """
    n_points = data_matrix.shape[0]

    n_points, n_dim = data_matrix.shape
    point_indices = np.arange(start=0, stop=n_points)
    generated_points = np.zeros(shape=(k, n_dim))
    
    for i in range(k):
        # Generate a random vector
        random_vector = np.random.rand(n_samples, 1)

        # Scale the random vector to L1 unit norm
        proba_vector = random_vector / random_vector.sum()

        # Select data points
        if n_samples is None or n_samples == n_points:
            selected_indices = point_indices
        else:
            selected_indices = np.random.choice(point_indices, n_samples)

        # Generate a point by taking the convex combination of the selected input points.
        if issparse(data_matrix):
            new_point = safe_sparse_dot(proba_vector.T, data_matrix[selected_indices])
        else:
            new_point = np.dot(proba_vector.T, data_matrix[selected_indices])

        generated_points[i] = new_point

    return generated_points


def compute_real_dataset_costs(run_info: RunInfo, coreset_path: Path, n_candidate_solutions: int, synthetic_center_type: str=None) -> None:
    experiment_dir = coreset_path.parent

    if synthetic_center_type == "meb":
        file_postfix = "_synthetic"
    elif synthetic_center_type == "convex":
        file_postfix = "_convexsynthetic"
    else:
        file_postfix = ""

    coreset_cost_path = experiment_dir / f"real_cost{file_postfix}.txt"
    real_cost_path = experiment_dir / f"real_cost{file_postfix}.txt"
    
    if coreset_cost_path.exists() and real_cost_path.exists():
        print("Costs are exists! Existing...")
        return

    added_cost = get_added_cost_for_low_dimensional_dataset(run_info)

    unzipped_result_path = unzip_file(coreset_path)
    original_data_points = load_original_data(run_info)

    coreset = np.loadtxt(fname=unzipped_result_path, dtype=np.double, delimiter=' ', skiprows=1)
    # coreset_weights = coreset[:,0]
    coreset_points = coreset[:,1:]

    # Generate a number of candidate solutions and compute their costs
    for solution_index in range(1, n_candidate_solutions+1):
        print(f"Generating solution #{solution_index}...")
        solution_path = coreset_path.parent / f"centers{file_postfix}.txt"
        if synthetic_center_type == "meb":
            centers = generate_random_points_within_minimum_enclosing_ball(data_matrix=coreset_points, k=run_info.k)
            np.savetxt(str(solution_path), centers)
        elif synthetic_center_type == "convex":
            kmeans_centers = get_centers(unzipped_result_path)
            centers = generate_random_points_within_convex_hull(data_matrix=kmeans_centers, k=run_info.k)
            np.savetxt(str(solution_path), centers)
        else:
            centers = get_centers(unzipped_result_path)

        coreset_cost_path = compute_coreset_costs(
            coreset=coreset,
            center_points=centers,
            added_cost=added_cost,
            cost_file_path=experiment_dir / f"coreset_cost{file_postfix}.txt",
        )

        if run_info.dataset == "nytimespcalowd":
            centers = expand_dimensionality_nytimes(k=run_info.k, centers=centers)

        real_cost_path = compute_real_cost(
            data_points=original_data_points,
            center_points=centers,
            cost_file_path=experiment_dir / f"real_cost{file_postfix}.txt",
        )
        centers = None
        del centers

        for src_path in [coreset_cost_path, real_cost_path, solution_path]:
            # Assume all file paths have .txt extention
            dest_path = src_path.parent / f"solution{solution_index}-{src_path.name}"
            shutil.move(str(src_path), dest_path)

    print(f"Successfully generate candidate solutions. Removing {unzipped_result_path}...")
    os.remove(unzipped_result_path)

    # Compute distortions of candidate solutions.
    df_solution_distortions = collect_distortions_of_solutions(
        costs_dir=experiment_dir,
        n_candidate_solutions=n_candidate_solutions,
        file_postfix=file_postfix,
    )
    print(f"Solution distortions:\n{df_solution_distortions}")

    # Find worst-case candidate solution.
    max_distortion_idx = df_solution_distortions['distortion'].idxmax()
    max_distortion_row = df_solution_distortions.iloc[max_distortion_idx]
    max_distortion_sol_idx = max_distortion_row["solution_index"]

    # Rename file paths for worst-case solution.
    dest_files = [f"coreset_cost{file_postfix}.txt", f"real_cost{file_postfix}.txt", f"centers{file_postfix}.txt"]
    for file_name in dest_files:
        src_path = experiment_dir / f"solution{max_distortion_sol_idx}-{file_name}"
        dest_path = src_path.parent / file_name
        shutil.move(src_path, dest_path)


@click.command(help="Compute costs for coresets.")
@click.option(
    "-r",
    "--results-dir",
    type=click.STRING,
    required=True,
)
def main(results_dir: str) -> None:
    output_paths = find_unprocesses_result_files(results_dir)
    total_files = len(output_paths)
    for index, result_path in enumerate(output_paths):
        print(f"Processing file {index+1} of {total_files}: {result_path}")
        experiment_dir = result_path.parent

        run_info = load_run_info(experiment_dir)
        if run_info is None:
            print("Cannot process results file because run file is missing.")
            continue

        if not os.path.exists(run_info.dataset_path):
            print(f"Dataset path: {run_info.dataset_path} cannot be found. Skipping...")
            continue

        if "hardinstance" in run_info.dataset:
            # Algorithms on the benchmark dataset are evaluated differently.
            compute_benchmark_costs(run_info=run_info, coreset_path=result_path)
        else:
            compute_real_dataset_costs(
                run_info=run_info,
                coreset_path=result_path,
                n_candidate_solutions=5,
                synthetic_center_type=None,
            )
            # compute_real_dataset_costs(
            #     run_info=run_info,
            #     coreset_path=result_path,
            #     n_candidate_solutions=5,
            #     synthetic_center_type="meb",
            # )
            compute_real_dataset_costs(
                run_info=run_info,
                coreset_path=result_path,
                n_candidate_solutions=5,
                synthetic_center_type="convex",
            )

        print(f"Done processing file {index+1} of {total_files}.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
