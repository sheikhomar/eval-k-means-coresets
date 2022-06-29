#ifndef PARSERS_H_INCLUDED
#define PARSERS_H_INCLUDED


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
#include "stop_watch.h"


template <typename DataMatrixType>
void parseBoW(const std::string &filePath, DataMatrixType &data, bool transposed = false)
{
    printf("Opening input file %s...\n", filePath.c_str());
    namespace io = boost::iostreams;

    std::ifstream fileStream(filePath, std::ios_base::in | std::ios_base::binary);
    io::filtering_streambuf<io::input> filteredInputStream;
    if (boost::ends_with(filePath, ".gz"))
    {
        filteredInputStream.push(io::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    // The format of the BoW files is 3 header lines, followed by data triples:
    // ---
    // D    -> the number of documents
    // W    -> the number of words in the vocabulary
    // NNZ  -> the number of nonzero counts in the bag-of-words
    // docID wordID count
    // docID wordID count
    // ...
    // docID wordID count
    // docID wordID count
    // ---

    std::string line;
    std::getline(inData, line); // Read line with D
    auto dataSize = std::stoul(line.c_str());
    std::getline(inData, line); // Read line with W
    auto dimSize = std::stoul(line.c_str());
    std::getline(inData, line); // Skip line with NNZ

    printf("Data size: %ld, vocabulary size: %ld\n", dataSize, dimSize);

    bool firstDataLine = true;
    size_t previousDocId = 0, currentRow = 0, docId = 0, wordId = 0;
    size_t lineNo = 3;
    double count;

    if (transposed)
    {
        data.allocate(dimSize, dataSize);
    }
    else
    {
        data.allocate(dataSize, dimSize);
    }

    while (inData.good())
    {
        std::getline(inData, line);
        lineNo++;

        std::vector<std::string> splits;
        boost::split(splits, line, boost::is_any_of(" "));

        if (splits.size() != 3)
        {
            printf("Skipping line no %ld: '%s'.\n", lineNo, line.c_str());
            continue;
        }

        docId = std::stoul(splits[0]);
        wordId = std::stoul(splits[1]) - 1; // Convert to zero-based array indexing
        count = static_cast<double>(std::stoul(splits[2]));

        if (firstDataLine)
        {
            firstDataLine = false;
            previousDocId = docId;
        }

        if (previousDocId != docId)
        {
            currentRow++;
        }

        if (transposed)
        {
            data.set(wordId, currentRow, count);
        }
        else
        {
            data.set(currentRow, wordId, count);
        }

        previousDocId = docId;
    }
}


template <typename DataMatrixType>
void parseSparseBoW(const std::string &filePath, DataMatrixType &data, bool transposed = false)
{
    printf("Opening input file %s...\n", filePath.c_str());
    namespace io = boost::iostreams;

    std::ifstream fileStream(filePath, std::ios_base::in | std::ios_base::binary);
    io::filtering_streambuf<io::input> filteredInputStream;
    if (boost::ends_with(filePath, ".gz"))
    {
        filteredInputStream.push(io::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    // The format of the BoW files is 3 header lines, followed by data triples:
    // ---
    // D    -> the number of documents
    // W    -> the number of words in the vocabulary
    // NNZ  -> the number of nonzero counts in the bag-of-words
    // docID wordID count
    // docID wordID count
    // ...
    // docID wordID count
    // docID wordID count
    // ---

    std::string line;
    std::getline(inData, line); // Read line with D
    auto dataSize = std::stoul(line.c_str());
    std::getline(inData, line); // Read line with W
    auto dimSize = std::stoul(line.c_str());
    std::getline(inData, line); // Skip line with NNZ
    auto nnz = std::stoul(line.c_str());

    printf("Data size: %ld, vocabulary size: %ld, NNZ: %ld\n", dataSize, dimSize, nnz);

    bool firstDataLine = true;
    size_t previousDocId = 0, currentRow = 0, docId = 0, wordId = 0;
    size_t lineNo = 3;
    double count;

    if (transposed)
    {
        data.setSize(dimSize, dataSize);
    }
    else
    {
        data.setSize(dataSize, dimSize);
    }

    StopWatch sw(true);

    while (inData.good())
    {
        std::getline(inData, line);
        lineNo++;

        std::vector<std::string> splits;
        boost::split(splits, line, boost::is_any_of(" "));

        if (splits.size() != 3)
        {
            printf("Skipping line no %ld: '%s'.\n", lineNo, line.c_str());
            continue;
        }

        docId = std::stoul(splits[0]);
        wordId = std::stoul(splits[1]) - 1; // Convert to zero-based array indexing
        count = static_cast<double>(std::stoul(splits[2]));

        if (firstDataLine)
        {
            firstDataLine = false;
            previousDocId = docId;
        }

        if (previousDocId != docId)
        {
            currentRow++;
        }

        if (transposed)
        {
            data.set(wordId, currentRow, count);
        }
        else
        {
            data.set(currentRow, wordId, count);
        }

        previousDocId = docId;
    }

    std::cout << "Data parsed in " << sw.elapsedStr() << std::endl;
}

#endif