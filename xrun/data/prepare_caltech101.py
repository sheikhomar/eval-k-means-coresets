import os, requests, subprocess, shutil, gzip

from pathlib import Path
from timeit import default_timer as timer
from typing import List

import click
import cv2
import h5py
import numpy as np
import pandas as pd

from tqdm.std import tqdm


def unpack_dataset_file(input_path: Path) -> Path:
    dataset_dir = input_path.parent
    print(f"Unpacking file {input_path}...")
    extract_dir = str(dataset_dir)
    p = subprocess.Popen(
        args=[
            "tar",
            "-xf",
            os.path.abspath(input_path),
            "101_ObjectCategories",
        ],
        start_new_session=False,
        cwd=extract_dir,
    )
    p.wait()
    return dataset_dir / "101_ObjectCategories"


def store_disc(sift_features: np.ndarray, output_path: Path) -> None:
    print("Storing data on disk...")
    start_time = timer()
    df_data = pd.DataFrame(sift_features)
    df_data.to_csv(output_path, index=False, header=False)
    end_time = timer()
    print(f"Elapsed time: {end_time - start_time:.2f} secs")


def extract_sift_features(file_paths: List[Path]) -> None:
    sift_algo = cv2.SIFT_create()
    all_descriptors = []
    for file_path in tqdm(file_paths):
        img = cv2.imread(str(file_path))
        sift_keypoints, sift_desc = sift_algo.detectAndCompute(img, None)
        all_descriptors.append(sift_desc.astype(np.int16))
    combined_arr = np.concatenate(all_descriptors, axis=0)
    print(f"Generated matrix shape: {combined_arr.shape}")
    return combined_arr


def prepare_caltech101() -> None:
    local_file_path = Path("data/input/caltech101-sift.txt.gz")

    if not local_file_path.parent.exists():
        os.makedirs(str(local_file_path.parent))

    if not local_file_path.exists():
        tar_file_path = Path("data/input/101_ObjectCategories.tar.gz")
        if not tar_file_path.exists():
            raise Exception(f"Caltech101 dataset cannot be found.\nPlease download it manually from https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp&export=download into {tar_file_path}.")
        dataset_dir = unpack_dataset_file(tar_file_path)
        print(f"Unpacked to {dataset_dir}")
        file_paths = dataset_dir.glob("**/*.jpg")
        # Exclude directory 'BACKGROUND_Google'
        file_paths = filter(lambda f:  f.parent.name != "BACKGROUND_Google", file_paths)
        file_paths = list(sorted(file_paths))
        sift_features = extract_sift_features(file_paths)
        store_disc(sift_features, local_file_path)


@click.command(help="Extracts SIFT features from Caltech101 dataset.")
def main():
    prepare_caltech101()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
