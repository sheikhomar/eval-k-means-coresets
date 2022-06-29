import os, requests, subprocess, shutil, gzip

from pathlib import Path
from timeit import default_timer as timer

import click
import h5py
import numpy as np


from sklearn.decomposition import TruncatedSVD, PCA
from tqdm.std import tqdm


def download_file(url: str, file_path: Path):
    """
    Downloads file from `url` to `file_path`.
    """
    print(f"Downloading {url} to {file_path}...")
    chunk_size = 1024
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        total_size = int(r.headers.get('Content-Length', 10 * chunk_size))
        pbar = tqdm( unit="B", unit_scale=True, total=total_size)
        for chunk in r.iter_content(chunk_size=chunk_size): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)


def unpack_dataset_file(input_path: Path) -> Path:
    dataset_file = input_path.parent / "SIFT10M" / "SIFT10Mfeatures.mat"
    if not dataset_file.exists():
        print(f"Unpacking file {input_path}...")
        extract_dir = os.path.abspath(str(input_path.parent))
        p = subprocess.Popen(
            args=[
                "tar",
                "-xvf",
                os.path.abspath(input_path),
                "SIFT10M/SIFT10Mfeatures.mat",
                "-C",
                extract_dir,
            ],
            start_new_session=False,
            cwd=extract_dir,
        )
        p.wait()
    return dataset_file


def download_and_extract(local_file_path) -> Path:
    download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00353/SIFT10M.tar.gz"
    tar_file = Path("data/input/sift10m.tar.gz")
    if not tar_file.exists():
        download_file(download_url, tar_file)
    dataset_file = unpack_dataset_file(tar_file)
    return dataset_file


def convert_hdf5_to_csv(hdf5_file_path: Path, csv_file_path: Path) -> None:
    hdf5_fp = h5py.File(str(hdf5_file_path))
    hdf5_dataset = hdf5_fp['fea']
    print(f"Converting HDF5 dataset of shape {hdf5_dataset.shape} to Numpy...")
    numpy_dataset = np.array(hdf5_dataset)
    print(f"Saving data into CSV: {csv_file_path}...")
    np.savetxt(fname=csv_file_path, X=numpy_dataset, delimiter=",")
    print("Done")


def ensure_dataset_exists():
    local_file_path = Path("data/input/sift10m.txt.gz")
    # expected_file_size = 19138633

    if not local_file_path.parent.exists():
        os.makedirs(str(local_file_path.parent))

    # if local_file_path.exists():
    #     actual_file_size = local_file_path.stat().st_size
    #     if actual_file_size < expected_file_size:
    #         print(f"The size of file {local_file_path.name} is {actual_file_size} but expected {expected_file_size}. Removing file...")
    #         os.remove(local_file_path)
    
    if not local_file_path.exists():
        dataset_file = download_and_extract(local_file_path)
        convert_hdf5_to_csv(dataset_file, local_file_path)


def prepare_sift10m() -> None:
    ensure_dataset_exists()


@click.command(help="Download and convert SIFT10M dataset.")
def main():
    prepare_sift10m()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
