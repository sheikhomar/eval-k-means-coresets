# An Empirical Evaluation of $k$-Means Coresets

## Getting Started

### Build C++ programs

```bash
# Install the prerequisite libraries and tools
./install_prerequisites.sh

# Install Python libraries
poetry install

# Build BICO.
make -C bico/build

# Build random seed generator.
make -C mt

# Build random projection program.
make -C rp

# Build k-means tool.
make -C kmeans

# Build main project.
cmake -S gs -B gs/build -G "Ninja"
cmake --build gs/build

```

### Datasets

Generate the `nytimes100d` dataset:

```bash
# Download file
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz \
    -O data/input/docword.nytimes.txt.gz
# Perform dimensionality reduction via random projection.
export CPATH=/path/to/boost_1_76_0
export LIBRARY_PATH=/path/to/boost_1_76_0/stage/lib
make -C rp && rp/bin/rp.exe \
    reduce-dim \
    data/input/docword.nytimes.txt.gz \
    8192,100 \
    0 \
    1704100552 \
    data/input/docword.nytimes.rp8192-100.txt.gz
```

Generate the `nytimespcalowd` dataset:

```bash
poetry run python -m xrun.data.tsvd -i data/input/docword.nytimes.txt.gz -d 10,20,30,40,50
```

## Run Experiments

```bash
poetry run python -m xrun.go
```

## Reference

```code
@inproceedings{schwiegelshohn2022empirical,
  title={An Empirical Evaluation of $k$-Means Coresets},
  author={Schwiegelshohn, Chris and Sheikh-Omar, Omar},
  booktitle={European Symposium on Algorithms},
  year={2022}
}
```
