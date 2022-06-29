import os

from pathlib import Path

from timeit import default_timer as timer
from typing import List, Optional

import numpy as np
import click

from sklearn.cluster import KMeans

from xrun.data.loader import load_dataset


@click.command(help="Generates cluster curse.")
@click.option(
    "-i",
    "--input-data-path",
    type=click.STRING,
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.STRING,
    required=True,
)
def main(input_data_path: str, output_path: str) -> None:
    X = load_dataset(input_data_path)

    k_ranges = [10, 20, 30, 40, 50, 70, 100, 200]

    for k in k_ranges:
        print(f"Computing k-Means clustering for k={k}")
        start_time = timer()
        algo = KMeans(n_clusters=k, verbose=0)
        algo.fit(X)
        cost = algo.inertia_
        running_time = timer() - start_time
        print(f"  Completed in  {timer() - start_time} seconds... Final cost: {cost}")
        with open(output_path, "a") as file:
            file.write(f"{input_data_path},{k},{cost},{running_time}\n")



if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
