import shutil, gzip

from pathlib import Path
from pprint import pprint

import click

from tqdm.std import tqdm


def process_file(file_path: Path):
    print(f"Processing {file_path}...")
    # Make a copy of the input file.
    old_file_path = Path(file_path.parent / f"old-{file_path.name}")
    shutil.move(file_path, old_file_path)

    valid_lines = []
    num_of_nans_found = 0
    with gzip.open(str(old_file_path), "rt") as in_file:
        for i, line in enumerate(in_file):
            if i == 0 or len(line) == 0:
                # Skip the first line as it contains the number of lines in the original file.
                # Skip any empty lines.
                continue
            if 'nan' in line:
                num_of_nans_found += 1
            if 'nan' not in line:
                valid_lines.append(line)
    
    if num_of_nans_found > 0:
        print(f" - Found {num_of_nans_found} lines with NaN entries.")
        # Only write out the filtered lines to disk if NaNs are found.
        with gzip.open(str(file_path), "wt") as out_file:
            out_file.write(f"{len(valid_lines)}\n")  # Write number of lines
            out_file.writelines(valid_lines) # Write all lines except the first line
    else:
        shutil.move(old_file_path, file_path)


def remove_nans(input_path: str) -> None:
    file_paths = list(Path(input_path).glob("**/**/*.txt.gz"))
    print(f"Found {len(file_paths)} files to process.")
    for file_path in file_paths:
        process_file(file_path)


@click.command(help="Removes lines with NaN from results.txt.gz files.")
@click.option(
    "-i",
    "--input-path",
    type=click.STRING,
    required=True,
)
def main(input_path: str):
    remove_nans(
        input_path=input_path,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
