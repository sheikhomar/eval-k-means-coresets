import os, requests, subprocess, json, shutil, time

from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional

import click
import psutil

from tqdm import tqdm

from xrun.data.run_info import RunInfo

class Dataset:
    _name: str

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_local_file_path(self, k: int) -> Path:
        raise Exception("Not implemented")

    def get_file_size(self, k: int) -> int:
        raise Exception("Not implemented")

    def create_local_file(self, k: int) -> None:
        raise Exception("Not implemented")

    def ensure_exists(self, k: int):
        local_file_path = self.get_local_file_path(k)

        if not local_file_path.parent.exists():
            os.makedirs(str(local_file_path.parent))

        if local_file_path.exists():
            actual_file_size = local_file_path.stat().st_size
            expected_file_size = self.get_file_size(k)
            if actual_file_size < expected_file_size:
                print(f"The size of file {local_file_path.name} is {actual_file_size} but expected {expected_file_size}.")
        
        if not local_file_path.exists():
            self.create_local_file(k)


class ExternalDataset(Dataset):
    def __init__(self, name: str, download_url: str, file_size: int) -> None:
        super().__init__(name)
        self._download_url = download_url
        self._file_size = file_size
        self._local_file_name = os.path.basename(download_url)

    @property
    def download_url(self) -> str:
        return self._download_url

    @property
    def local_file_name(self) -> str:
        return self._local_file_name

    @property
    def local_file_path(self) -> Path:
        return self.get_local_file_path(0)

    def get_local_file_path(self, k: int) -> Path:
        return Path(f"data/input/{self._local_file_name}")

    def get_file_size(self, k: int) -> int:
        return self._file_size

    def create_local_file(self, k: int) -> None:
        local_file_path = self.get_local_file_path(k)
        self._download_file(url=self._download_url, file_path=local_file_path)

    def _download_file(self, url: str, file_path: Path):
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


class BenchmarkDataset(Dataset):
    def __init__(self, beta: int) -> None:
        super().__init__(name="hardinstance")
        self._beta = beta
    
    def get_local_file_path(self, k: int) -> Path:
        names = {
            10: f"benchmark-k10-alpha6-beta{self._beta:0.2f}.txt.gz",
            20: f"benchmark-k20-alpha5-beta{self._beta:0.2f}.txt.gz",
            30: f"benchmark-k30-alpha4-beta{self._beta:0.2f}.txt.gz",
            40: f"benchmark-k40-alpha4-beta{self._beta:0.2f}.txt.gz",
            50: f"benchmark-k50-alpha4-beta{self._beta:0.2f}.txt.gz",
        }
        return Path(f"data/input/{names[k]}")

    def get_file_size(self, k: int) -> int:
        sizes = {
            10: 3357717,
            20: 13710452,
            30: 9356322,
            40: 17663742,
            50: 44040300,
        }
        return sizes[k]

    def create_local_file(self, k: int) -> None:
        local_file_path = self.get_local_file_path(k)
        raise Exception(f"Cannot create {local_file_path}! Run `python -m xrun.data.gen_benchmark`")


class LowDimensionalDataset(Dataset):
    def __init__(self, original_dataset: ExternalDataset) -> None:
        super().__init__(original_dataset.name)
        self._original_dataset = original_dataset

    def get_local_file_path(self, k: int) -> Path:
        file_path = self._original_dataset.get_local_file_path(k)
        return Path(f"{file_path}-svd-d{k}.txt.gz")
    
    def get_file_size(self, k: int) -> int:
        sizes_dict = {
            "census": {
                10: 1529828319,
                20: 1533766277,
                30: 1532138499,
                40: 1527974659,
                50: 1523368090,
            },
            "covertype": {
                10: 290859471,
                20: 289207596,
                30: 286757020,
                40: 286788280,
                50: 286788280,
            },
            "caltech101": {
                10: 4202925391,
                20: 4215519948,
                30: 4220988256,
                40: 4223419818,
                50: 4224802601,
            },
            "nytimes": {
                10: 4202925391,
                20: 4215519948,
                30: 4220988256,
                40: 4223419818,
                50: 4224802601,
            },
        }
        return sizes_dict[self._original_dataset.name][k]

    def create_local_file(self, k: int) -> None:
        local_file_path = self.get_local_file_path(k)
        original_file_path = self._original_dataset.get_local_file_path(k)
        raise Exception(f"Cannot create {local_file_path}! Run `python -m xrun.data.tsvd -i {original_file_path} -d {k}`")


class LocalDataset(Dataset):
    def __init__(self, name: str, file_path: str, file_size: int) -> None:
        super().__init__(name)
        self._file_path = Path(file_path)
        self._file_size = file_size

    def get_local_file_path(self, k: int) -> Path:
        return self._file_path
    
    def get_file_size(self, k: int) -> int:
        return self._file_size

    def create_local_file(self, k: int) -> None:
        local_file_path = self.get_local_file_path(k)
        raise Exception(f"Cannot create {local_file_path}!")


class ExperimentRunner:
    _datasets : Dict[str, Dataset] = {
        "census": ExternalDataset(
                    name="census",
                    download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt",
                    file_size=361344227
                ),
        "covertype": ExternalDataset(
                    name="covertype",
                    download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
                    file_size=11240707
                ),
        "enron": ExternalDataset(
                    name="enron",
                    download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz",
                    file_size=12313965
                ),
        # "pubmed": ExternalDataset(
        #             name="pubmed",
        #             download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.pubmed.txt.gz",
        #             file_size=1878648166
        #         ),
        "nytimes": ExternalDataset(
                    name="nytimes",
                    download_url="https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz",
                    file_size=234225967
                ),
        "tower": ExternalDataset(
                    name="tower",
                    download_url="http://homepages.uni-paderborn.de/frahling/instances/Tower.txt",
                    file_size=52828754
                ),
        "hardinstanceb1": BenchmarkDataset(beta=1),
        "hardinstanceb15": BenchmarkDataset(beta=1.5),
        "hardinstanceb2": BenchmarkDataset(beta=2),
        "caltech101": LocalDataset("caltech101", "data/input/caltech101-sift.txt.gz", 100000),
        "nytimes100d": LocalDataset("nytimes100d", "data/input/docword.nytimes.rp8192-100.txt.gz", 35071691),
    }
    _datasets["censuslowd"] = LowDimensionalDataset(_datasets["census"])
    _datasets["covertypelowd"] = LowDimensionalDataset(_datasets["covertype"])
    _datasets["caltech101lowd"] = LowDimensionalDataset(_datasets["caltech101"])
    _datasets["nytimespcalowd"] = LowDimensionalDataset(_datasets["nytimes"])
    _dir_ready = "data/queue/ready"
    _dir_in_progress = "data/queue/in-progress"
    _dir_completed = "data/queue/completed"
    _dir_discarded = "data/queue/discarded"
    _dir_output = None
    _child_processes: List[subprocess.Popen] = []

    def __init__(self, output_dir: str) -> None:
        self._dir_output = output_dir
        for directory in [self._dir_ready, self._dir_in_progress, self._dir_completed, self._dir_discarded]:
            if not os.path.exists(directory):
                print(f"Creating directory {directory}...")
                os.makedirs(directory)

    def run(self, max_active: int) -> None:
        while True:
            self._clean_in_progress()

            n_active = len(self._find_in_progress_files())
            
            if n_active < max_active:
                self._lunch_new_run()
        
            # print("Checking CPU utilization: ", end="")
            # cpu_utilisation = psutil.cpu_percent(interval=1, percpu=False)
            # print(f"{cpu_utilisation} %")

            time.sleep(2)

    def _clean_in_progress(self):
        file_paths = self._find_in_progress_files()
        for file_path in file_paths:
            run = RunInfo.load_json(file_path)
            if not self._is_running(run.process_id):
                print(f"Process {run.process_id} which started {run.started_at} is not running anymore.")

                # Check if the result file is created.
                done_file_path = Path(run.output_dir) / "done.out"
                if done_file_path.exists():
                    completed_at = datetime.fromtimestamp(done_file_path.stat().st_ctime)
                    print(f" - Completed at {completed_at}. Moving to completed.")
                    run.end_time = completed_at.isoformat()
                    run.duration_secs = (completed_at - run.started_at).total_seconds()
                    run.process_id = -2
                    run.save_json(file_path)
                    shutil.move(file_path, f"{self._dir_completed}/{file_path.name}")
                    shutil.copy(src=f"{self._dir_completed}/{file_path.name}", dst=run.output_dir)
                else:
                    print(" - Process stopped but done.out file does not exist! Discarding run.")
                    self._move_to_discarded(file_path)

    def _is_running(self, process_id: int) -> bool:
        for proc in self._child_processes:
            if proc.pid == process_id:
                # poll() checks the process has terminated. Returns None value 
                # if process has not terminated yet.
                return proc.poll() is None

        # psutil cannot detect child processes.
        for proc in psutil.process_iter():
            if process_id == proc.pid:
                return True
        return False

    def _find_in_progress_files(self) -> List[Path]:
        return self._find_json_files(self._dir_in_progress)

    def _lunch_new_run(self) -> None:
        # Find the next run file containing the experiment to run
        while True:
            run_file_path = self._get_next_run_file()
            if run_file_path is None:
                print("Ran out of experiments to run!")
                return
            if self._should_discard(run_file_path):
                self._move_to_discarded(run_file_path)
            else:
                print(f"Will execute experiment from file {run_file_path}")
                break

        # Prepare for launch
        run_file_path = self._move_to_progress(run_file_path)
        run_details: RunInfo = RunInfo.load_json(run_file_path)
        experiment_dir = self._get_experiment_dir(run_details)
        cmd = self._build_command(run_details, experiment_dir)

        # Actual launch
        print(f"Launching experiment with command:\n '{cmd}'")
        p = subprocess.Popen(
            args=cmd,
            stdout=open(experiment_dir / "stdout.out", "a"),
            stderr=open(experiment_dir / "stderr.out", "a"),
            start_new_session=True
        )
        self._child_processes.append(p)

        # Create PID file.
        with open(experiment_dir / "pid.out", "w") as f:
            f.write(str(p.pid))

        # Persist run details to disk.
        run_details.output_dir = str(experiment_dir)
        run_details.command = " ".join(cmd)
        run_details.start_time = datetime.now().isoformat()
        run_details.process_id = p.pid
        run_details.save_json(run_file_path)

    def _build_command(self, run: RunInfo, experiment_dir: Path) -> List[str]:
        dataset_name = run.dataset
        dataset = self._datasets[dataset_name]
        dataset.ensure_exists(run.k)
        data_file_path = dataset.get_local_file_path(run.k)

        if run.algorithm == "bico":
            algorithm_exe_path = "bico/bin/BICO_Quickstart.exe"
            cmd = [
                algorithm_exe_path,
                dataset_name, # Dataset
                str(data_file_path), # Input path
                str(run.k), # Number of clusters
                str(run.m), # Coreset size
                str(run.randomSeed), # Random Seed
                str(experiment_dir), # Output dir
            ]
            return cmd
        else:
            algorithm_exe_path = "gs/build/gs"
            cmd = [
                algorithm_exe_path,
                run.algorithm,
                dataset_name, # Dataset
                str(data_file_path), # Input path
                str(run.k), # Number of clusters
                str(run.m), # Coreset size
                str(run.randomSeed), # Random Seed
                str(experiment_dir), # Output dir
            ]
            return cmd

    def _get_experiment_dir(self, run: RunInfo) -> Path:
        experiment_no = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        experiment_dir = os.path.join(self._dir_output, f"{run.dataset}/{run.algorithm}-k{run.k}-m{run.m}/{experiment_no}")
        os.makedirs(experiment_dir)
        return Path(experiment_dir)

    def _move_to_progress(self, run_file_path: Path) -> Path:
        inprogress_path = f"{self._dir_in_progress}/{run_file_path.name}"
        shutil.move(run_file_path, inprogress_path)
        return inprogress_path

    def _should_discard(self, run_file_path: Path) -> bool:
        paths_to_check = [
            os.path.join(self._dir_in_progress, run_file_path.name),
            os.path.join(self._dir_completed  , run_file_path.name),
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                return True
        return False

    def _move_to_discarded(self, run_file_path: Path) -> None:
        discarded_file_name = run_file_path.name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        discarded_path = f"{self._dir_discarded}/{discarded_file_name}"
        shutil.move(run_file_path, discarded_path)
        print(f"Experiment {run_file_path.name} is in progress or completed. Moving to {discarded_path}...")

    def _get_next_run_file(self) -> Optional[Path]:
        file_paths = self._find_json_files(self._dir_ready)
        if len(file_paths) > 0:
            sorted_file_paths = list(sorted(file_paths, key=lambda file_path: file_path.name))
            return sorted_file_paths[0]
        return None
    
    def _find_json_files(self, dir_name: str) -> List[Path]:
        dir_name_str = str(dir_name)
        return [
            Path(f"{dir_name_str}/{file_name}")
            for file_name in os.listdir(dir_name_str)
            if file_name.endswith(".json")
        ]

@click.command(help="Run experiments.")
@click.option(
    "-o",
    "--output-dir",
    type=click.STRING,
    required=True,
)
@click.option(
    "-m",
    "--max-active",
    type=click.INT,
    default=1,
    help="Maximum number of simultaneous runs."
)
def main(output_dir: str, max_active: int):
    if not os.path.exists(output_dir):
        print(f"The directory {output_dir} does not exist. Please provide an existing directory.")
        return
    ExperimentRunner(output_dir=output_dir).run(max_active=max_active)

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
