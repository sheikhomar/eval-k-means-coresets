import dataclasses
import json

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunInfo:
    algorithm: str
    dataset: str
    k: int
    m: int
    iteration: int
    randomSeed: int
    output_dir: str
    command: str
    start_time: str
    end_time: str
    duration_secs: float
    process_id: int

    @classmethod
    def load_json(cls, file_path: Path):
        with open(file_path, "r") as f:
            content = json.load(f)
            obj = cls(
                algorithm=content["algorithm"],
                dataset=content["dataset"],
                k=content["k"],
                m=content["m"],
                iteration=content.get("iteration", -1),
                randomSeed=content["randomSeed"],
                output_dir=content.get("output_dir", ""),
                command=content.get("command", ""),
                start_time=content.get("start_time", ""),
                end_time=content.get("end_time", ""),
                duration_secs=content.get("duration_secs", 0),
                process_id=content.get("process_id", -1),
            )
            return obj

    def save_json(self, file_path: Path):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4, sort_keys=False)

    @property
    def started_at(self) -> datetime:
        return datetime.fromisoformat(self.start_time)

    @property
    def dataset_path(self) -> str:
        args = self.command.split(" ")
        dataset_path = args[2] if self.algorithm == "bico" else args[3]
        return dataset_path

    @property
    def is_low_dimensional_dataset(self) -> str:
        return "lowd" in self.dataset

    @property
    def original_dataset_name(self) -> str:
        if self.dataset == "nytimespcalowd":
            return "nytimes"
        return self.dataset.replace("lowd", "")
