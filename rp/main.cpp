#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <cstdlib>
#include <stdexcept>
#include <random>
#include <map>
#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include "random_engine.h"
#include "matrices.h"
#include "stop_watch.h"
#include "sketches.h"
#include "parsers.h"

template <typename MatrixType>
void printSquaredDistance(MatrixType &data, size_t p1, size_t p2)
{
    size_t D = data.rows();

    double squaredDistance = 0.0;
    double diff = 0.0;
    for (size_t j = 0; j < D; j++)
    {
        diff = data.at(j, p1) - data.at(j, p2);
        squaredDistance += diff * diff;
    }

    printf(" %10.2f ", squaredDistance);
}

template <typename MatrixType>
void printPairwiseSquaredDistances(MatrixType &data, int indices[], size_t numberOfSamples)
{
    auto printLine = [numberOfSamples]()
    {
        std::cout << "  ";
        for (size_t i = 0; i < numberOfSamples; i++)
        {
            std::cout << "------------";
        }
        std::cout << "\n";
    };

    std::cout << "Pairwise distances for points:\n ";
    for (size_t i = 0; i < numberOfSamples; i++)
    {
        printf("%12d", indices[i]);
    }
    std::cout << "\n";

    printLine();
    for (size_t i = 0; i < numberOfSamples; i++)
    {
        printf("  ");
        for (size_t j = 0; j < numberOfSamples; j++)
        {
            auto p1 = static_cast<size_t>(indices[i]);
            auto p2 = static_cast<size_t>(indices[j]);
            printSquaredDistance(data, p1, p2);
        }
        printf("\n");
    }
    printLine();
}

void testPairwiseSquaredDistances()
{
    Matrix data;
    data.allocate(2, 10);
    data.set(0, 0, -7.237310391208174);
    data.set(1, 0, -9.031086522545417);
    data.set(0, 1, -8.16550136087066);
    data.set(1, 1, -7.008504394784431);
    data.set(0, 2, -7.022668436942146);
    data.set(1, 2, -7.570412890908223);
    data.set(0, 3, -8.863943061317665);
    data.set(1, 3, -5.0532398146772355);
    data.set(0, 4, 0.08525185826796045);
    data.set(1, 4, 3.6452829679480585);
    data.set(0, 5, -0.794152276623841);
    data.set(1, 5, 2.104951171962879);
    data.set(0, 6, -1.340520809891421);
    data.set(1, 6, 4.157119493365752);
    data.set(0, 7, -10.32012970766661);
    data.set(1, 7, -4.33740290203162);
    data.set(0, 8, -2.187731658211975);
    data.set(1, 8, 3.333521246686991);
    data.set(0, 9, -8.535604566608127);
    data.set(1, 9, -6.013489256860859);

    int indices[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    printPairwiseSquaredDistances(data, indices, 10);

    // Expected result:
    //    0.00       4.95      2.18     18.47    214.31    165.53    208.7      31.53    178.38     10.79
    //    4.95       0.00      1.62      4.31    181.58    137.39    171.25     11.78    142.69      1.13
    //    2.18       1.62      0.00      9.73    176.31    132.41    169.82     21.33    142.27      4.71
    //   18.47       4.31      9.73      0.00    155.75    116.36    141.43      2.63    114.91      1.03
    //  214.31     181.58    176.31    155.75      0.00      3.15      2.29    172.        5.26    167.61
    //  165.53     137.39    132.41    116.36      3.15      0.00      4.51    132.25      3.45    125.84
    //  208.7      171.25    169.82    141.43      2.29      4.51      0.00    152.79      1.4     155.21
    //   31.53      11.78     21.33      2.63    172.      132.25    152.79      0.00    124.98      5.99
    //  178.38     142.69    142.27    114.91      5.26      3.45      1.4     124.98      0.00    127.66
    //   10.79       1.13      4.71      1.03    167.61    125.84    155.21      5.99    127.66      0.00
}

void runDense()
{
    Matrix data, sketch;

    std::string datasetName = "enron";
    std::string inputPath = "data/input/docword." + datasetName + ".txt.gz";
    std::string outputPath = "data/input/sketch-dense-docword." + datasetName + ".txt.gz";

    parseBoW("data/input/docword.enron.txt.gz", data, false);

    std::cout << "Running Clarkson Woodruff (CW) algorithm...\n";

    // Use Clarkson Woodruff (CW) algoritm reduce number of dimensions.
    sketch_cw(data, static_cast<size_t>(pow(2, 12)), sketch);

    std::cout << "Sketch generated!\n";

    std::cout << "Writing sketch to " << outputPath << std::endl;
    namespace io = boost::iostreams;
    std::ofstream fileStream(outputPath, std::ios_base::out | std::ios_base::binary);
    io::filtering_streambuf<io::output> fos;
    fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
    fos.push(fileStream);
    std::ostream outData(&fos);

    auto sketch_size = sketch.rows();
    outData << sketch_size << "\n";
    outData << sketch.columns() << "\n";
    outData << "0\n"; // Number of non-zero values is unknown
    double value;

    for (size_t i = 0; i < sketch.rows(); i++)
    {
        for (size_t j = 0; j < sketch.columns(); j++)
        {
            value = sketch.at(i, j);
            if (value != 0.0)
            {
                outData << (i + 1) << " " << (j + 1) << " " << value << "\n";
            }
        }
    }
}

void runSparseCsrMatrix()
{
    CooSparseMatrix cooData;
    size_t sketchSize = static_cast<size_t>(pow(2, 16));
    std::string datasetName = "nytimes";
    std::string inputPath = "data/input/docword." + datasetName + ".txt.gz";
    std::string outputPath = "data/input/sketch-docword." + datasetName + "." + std::to_string(sketchSize) + ".txt.gz";

    parseSparseBoW(inputPath, cooData);

    std::cout << "Data parsing completed!!\n";

    std::cout << "Running Clarkson Woodruff (CW) algorithm...\n";

    CsrMatrix csrData(cooData);
    std::vector<std::map<size_t, double>> sketch;

    sketch_cw_sparse(csrData, sketchSize, sketch);

    std::cout << "Sketch generated!\n";

    std::cout << "Writing sketch to " << outputPath << std::endl;
    namespace io = boost::iostreams;
    std::ofstream fileStream(outputPath, std::ios_base::out | std::ios_base::binary);
    io::filtering_streambuf<io::output> fos;
    fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
    fos.push(fileStream);
    std::ostream outData(&fos);

    auto sketch_size = sketch.size();
    outData << sketch_size << "\n";
    outData << cooData.columns() << "\n";
    outData << "0\n"; // Number of non-zero values is unknown
    size_t columnIndex;
    double value;

    for (size_t rowIndex = 0; rowIndex < sketch_size; rowIndex++)
    {
        for (auto &&pair : sketch[rowIndex])
        {
            columnIndex = pair.first;
            value = pair.second;
            if (value != 0.0)
            {
                outData << (rowIndex + 1) << " " << (columnIndex + 1) << " " << value << "\n";
            }
        }
    }
}

void toDense(const std::vector<std::map<size_t, double>> &sketch, size_t columns, Matrix &denseMatrix)
{
    denseMatrix.allocate(sketch.size(), columns);
    size_t columnIndex = 0UL;
    double value = 0.0;
    for (size_t rowIndex = 0; rowIndex < sketch.size(); rowIndex++)
    {
        for (auto &&pair : sketch[rowIndex])
        {
            columnIndex = pair.first;
            value = pair.second;
            denseMatrix.set(rowIndex, columnIndex, value);
        }
    }
}

void writeCsv(const Matrix &data, std::string &outputPath, bool transposed)
{
    std::cout << "Writing sketch to " << outputPath << std::endl;
    StopWatch sw(true);
    namespace io = boost::iostreams;
    std::ofstream fileStream(outputPath, std::ios_base::out | std::ios_base::binary);
    io::filtering_streambuf<io::output> fos;
    fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
    fos.push(fileStream);
    std::ostream outData(&fos);

    size_t dim1Size = transposed ? data.columns() : data.rows();
    size_t dim2Size = transposed ? data.rows() : data.columns();

    for (size_t i = 0; i < dim1Size; i++)
    {
        for (size_t j = 0; j < dim2Size; j++)
        {
            outData << (j > 0 ? ",": "");
            outData << (transposed ? data.at(j, i) : data.at(i, j));
        }
        outData << std::endl;
    }
    std::cout << " - Completed writing in " << sw.elapsedStr() << std::endl;
}

void writeCsv(const std::vector<std::map<size_t, double>> &data, size_t nColumns, std::string &outputPath)
{
    std::cout << "Writing sketch to " << outputPath << std::endl;
    StopWatch sw(true);
    namespace io = boost::iostreams;
    std::ofstream fileStream(outputPath, std::ios_base::out | std::ios_base::binary);
    io::filtering_streambuf<io::output> fos;
    fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
    fos.push(fileStream);
    std::ostream outData(&fos);

    auto sketch_size = data.size();
    outData << sketch_size << "\n";
    outData << nColumns << "\n";
    outData << "0\n"; // Number of non-zero values is unknown
    size_t columnIndex;
    double value;

    for (size_t rowIndex = 0; rowIndex < sketch_size; rowIndex++)
    {
        for (auto &&pair : data[rowIndex])
        {
            columnIndex = pair.first;
            value = pair.second;
            if (value != 0.0)
            {
                outData << (rowIndex + 1) << " " << (columnIndex + 1) << " " << value << "\n";
            }
        }
    }

    std::cout << " - Completed writing in " << sw.elapsedStr() << std::endl;
}

void reduceDimensions(std::string &inputPath, std::vector<size_t> sketchSizes, std::string &outputPath)
{
    if (sketchSizes.size() != 2)
    {
        throw std::out_of_range("Dimensionality reduction requires 2 sketch sizes.");
    }

    size_t cwSketchSize = sketchSizes[0];
    size_t radSketchSize = sketchSizes[1];

    CooSparseMatrix cooData;
    parseSparseBoW(inputPath, cooData, true);
    std::cout << "Data parsing completed! ";
    std::cout << "COO matrix ";
    std::cout << " rows:" << cooData.rows();
    std::cout << " columns=" << cooData.columns();
    std::cout << " nnz=" << cooData.nnz() << "\n";

    CsrMatrix csrData(cooData);
    std::cout << "Converted to CSR matrix:";
    std::cout << " rows=" << csrData.rows();
    std::cout << " columns=" << csrData.columns();
    std::cout << " nnz=" << csrData.nnz() << "\n";

    std::vector<std::map<size_t, double>> sketch;
    std::cout << "Running Clarkson Woodruff (CW) algorithm. Sketch size: " << cwSketchSize << " \n" << std::endl;
    sketch_cw_sparse(csrData, cwSketchSize, sketch);

    std::cout << "Sketch of size " << sketch.size() << " generated." << std::endl;

    Matrix cwSketch, radSketch;
    toDense(sketch, csrData.columns(), cwSketch);

    std::cout << "Dense sketch generated!" << sketch.size() << "\n";

    std::cout << "Running Rademacher (RAD) algorithm. Sketch size: " << radSketchSize << " \n" << std::endl;
    sketch_rad(cwSketch, radSketchSize, radSketch);

    std::cout << "Sketch generated!" << sketch.size() << "\n";

    writeCsv(radSketch, outputPath, true);
}

void sketchRowsClarksonWoodruff(std::string &inputPath, std::vector<size_t> sketchSizes, std::string &outputPath)
{
    if (sketchSizes.size() != 1)
    {
        throw std::out_of_range("Sketching rows requires one sketch size.");
    }

    size_t cwSketchSize = sketchSizes[0];

    CooSparseMatrix cooData;
    parseSparseBoW(inputPath, cooData);

    std::cout << "Running Clarkson Woodruff (CW) algorithm...\n";

    CsrMatrix csrData(cooData);
    std::vector<std::map<size_t, double>> sketch;
    sketch_cw_sparse(csrData, cwSketchSize, sketch);
    std::cout << "Sketch generated!\n";

    writeCsv(sketch, cooData.columns(), outputPath);
}

std::vector<size_t>
parseIntegers(std::string s, std::string delimiter = ",")
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<size_t> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
    {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(std::stoul(token));
    }

    res.push_back(std::stoul(s.substr(pos_start)));
    return res;
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cout << "Usage: algorithm dataset data_path k m seed output_path" << std::endl;
        std::cout << "  algorithm     = algorithm [cw, cw-rad]" << std::endl;
        std::cout << "  data_path     = file path to dataset" << std::endl;
        std::cout << "  sketch_sizes  = the size of the sketches" << std::endl;
        std::cout << "  sketch_rows   = 1 = sketch rows, 0 to sketch columns" << std::endl;
        std::cout << "  seed          = random seed" << std::endl;
        std::cout << "  output_path   = path to the results" << std::endl;
        std::cout << std::endl;
        std::cout << "6 arguments expected, got " << argc - 1 << ":" << std::endl;
        for (int i = 1; i < argc; ++i)
            std::cout << " " << i << ": " << argv[i] << std::endl;
        return 1;
    }

    std::string algorithmName(argv[1]);
    std::string dataFilePath(argv[2]);
    std::string sketchSizesStr(argv[3]);
    std::vector<size_t> sketchSizes = parseIntegers(sketchSizesStr);

    std::string sketchRowsStr(argv[4]);
    bool sketchRows = "1" == sketchRowsStr;
    int randomSeed = std::stoi(argv[5]);
    std::string outputPath(argv[6]);

    boost::algorithm::to_lower(algorithmName);
    boost::algorithm::trim(algorithmName);

    std::cout << "Running random projections with following parameters:\n";
    std::cout << " - Algorithm:     " << algorithmName << "\n";
    std::cout << " - Input path:    " << dataFilePath << "\n";
    std::cout << " - Sketch sizes:  ";
    for (size_t i = 0; i < sketchSizes.size(); i++)
    {
        std::cout << (i > 0 ? ", ": "") << sketchSizes[i];
    }
    std::cout << "\n";
    std::cout << " - Sketch rows:   " << sketchRows << "\n";
    std::cout << " - Random seed:   " << randomSeed << "\n";
    std::cout << " - Output path:    " << outputPath << "\n";

    if (randomSeed > 0)
    {
        RandomEngine::get().seed(5489UL); // Use fix seed.
    }
    else
    {
        /*
        Seeder produces uniformly-distributed unsigned integers with 32 bits of length.
        The entropy of the random_device may be lower than 32 bits.
        It is not a good idea to use std::random_device repeatedly as this may
        deplete the entropy in the system. It relies on system calls which makes it a very slow.
        Ref: https://diego.assencio.com/?index=6890b8c50169ef45b74db135063c227c
        */
        std::random_device seeder;
        RandomEngine::get().seed(seeder());
    }

    if (algorithmName == "reduce-dim")
    {
        reduceDimensions(dataFilePath, sketchSizes, outputPath);
    }
    else if (algorithmName == "sketch-cw")
    {
        sketchRowsClarksonWoodruff(dataFilePath, sketchSizes, outputPath);
    }
    else
    {
        std::cout << "Unknown algorithm: " << algorithmName << "\n";
        return -1;
    }

    return 0;
}
