#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <ctime>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <string>
#include <regex>
#include <stdexcept>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>

#include "../point/l2metric.h"
#include "../point/squaredl2metric.h"
#include "../point/point.h"
#include "../point/pointweightmodifier.h"
#include "../clustering/bico.h"
#include "../misc/randomness.h"
#include "../misc/randomgenerator.h"
#include "../misc/stopwatch.h"
#include "../datastructure/proxysolution.h"
#include "../point/pointcentroid.h"
#include "../point/pointweightmodifier.h"
#include "../point/realspaceprovider.h"

using namespace CluE;

class Experiment
{
protected:
    size_t DimSize;
    size_t DataSize;

public:
    size_t ClusterSize;
    size_t LowDimSize;
    size_t TargetCoresetSize;
    std::string InputFilePath;
    std::string OutputDir;

    void outputResultsToFile(ProxySolution<Point> *sol)
    {
        std::string outputFilePath = OutputDir + "/results.txt.gz";

        namespace io = boost::iostreams;
        std::ofstream fileStream(outputFilePath, std::ios_base::out | std::ios_base::binary);
        io::filtering_streambuf<io::output> fos;
        fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
        fos.push(fileStream);
        std::ostream outData(&fos);

        // Output coreset size
        outData << sol->proxysets[0].size() << "\n";

        // Output coreset points
        for (size_t i = 0; i < sol->proxysets[0].size(); ++i)
        {
            // Output weight
            outData << sol->proxysets[0][i].getWeight() << " ";
            // Output center of gravity
            for (size_t j = 0; j < sol->proxysets[0][i].dimension(); ++j)
            {
                outData << sol->proxysets[0][i][j];
                if (j < sol->proxysets[0][i].dimension() - 1)
                    outData << " ";
            }
            outData << "\n";
        }
    }

    void writeDoneFile()
    {
        std::string outputFilePath = OutputDir + "/done.out";
        std::ofstream outData(outputFilePath, std::ifstream::out);
        outData << "done\n";
        outData.close();
    }

    virtual void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        throw std::logic_error("parsePoint not yet implemented");
    }

    virtual void prepareFileStream(std::istream &inData)
    {
        throw std::logic_error("prepareFileStream not yet implemented");
    }

    void run()
    {
        printf("Opening input file %s...\n", InputFilePath.c_str());

        namespace io = boost::iostreams;
        std::ifstream fileStream(InputFilePath, std::ios_base::in | std::ios_base::binary);
        io::filtering_streambuf<io::input> filteredInputStream;
        if (boost::ends_with(InputFilePath, ".gz"))
        {
            filteredInputStream.push(io::gzip_decompressor());
        }
        filteredInputStream.push(fileStream);
        std::istream inData(&filteredInputStream);

        prepareFileStream(inData);

        std::string line;
        size_t pointCount = 0;
        StopWatch sw(true);
        Bico<Point> bico(DimSize, DataSize, ClusterSize, LowDimSize, TargetCoresetSize, new SquaredL2Metric(), new PointWeightModifier());

        while (inData.good())
        {
            std::vector<double> coords;
            parsePoint(coords, inData);
            CluE::Point p(coords);

            if (p.dimension() != DimSize)
            {
                std::clog << "Line skipped because line dimension is " << p.dimension() << " instead of " << DimSize << std::endl;
                continue;
            }

            pointCount++;

            if (pointCount % 10000 == 0)
            {
                std::cout << "Read " << pointCount << " points. Run time: " << sw.elapsedStr() << std::endl;
            }

            // Call BICO point update
            bico << p;

            // p.debugNonZero(pointCount, "%2.0f", 15);
            // p.debug(pointCount, "%5.0f", 15);
            // if (pointCount > 5) {
            //     break;
            // }
        }

        std::cout << "Processed " << pointCount << " points. Run time: " << sw.elapsedStr() << "s" << std::endl;

        outputResultsToFile(bico.compute());
        writeDoneFile();
    }
};

class CensusExperiment : public Experiment
{
public:
    CensusExperiment()
    {
        this->DimSize = 68UL;
        this->DataSize = 2458285UL;
        this->LowDimSize = 50UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        std::string line;
        std::getline(inData, line); // Ignore the header line.
        printf("Preparing Census Dataset. Skip first line: %s\n", line.c_str());
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        std::string line;
        std::getline(inData, line);

        std::vector<std::string> stringcoords;
        boost::split(stringcoords, line, boost::is_any_of(","));

        result.reserve(stringcoords.size());

        // Skip the first attribute which is `caseid`
        for (size_t i = 1; i < stringcoords.size(); ++i)
            result.push_back(atof(stringcoords[i].c_str()));
    }
};

class CovertypeExperiment : public Experiment
{
public:
    CovertypeExperiment()
    {
        this->DimSize = 54UL;
        this->DataSize = 581012UL;
        this->LowDimSize = 50UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Covertype.\n");
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        std::string line;
        std::getline(inData, line);

        std::vector<std::string> stringcoords;
        boost::split(stringcoords, line, boost::is_any_of(","));

        result.reserve(stringcoords.size());

        // Skip the last attribute because it is the label attribute
        // StreamKM++ paper removed the classification attribute so 
        // in total they have 54 attributes.
        for (size_t i = 0; i < stringcoords.size() - 1; ++i)
            result.push_back(atof(stringcoords[i].c_str()));
    }
};

class BagOfWordsExperiment : public Experiment
{
    std::string previousLine;
    size_t previousDocId;
    bool firstPoint;

public:
    BagOfWordsExperiment(size_t lowDimSize = 100UL)
    {
        this->LowDimSize = lowDimSize;
    }

    void prepareFileStream(std::istream &inData)
    {
        // The format of the docword.*.txt file is 3 header lines, followed by triples:
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
        this->DataSize = std::stoul(line.c_str());

        printf("Read D = %ld\n", this->DataSize);

        std::getline(inData, line); // Read line with W
        this->DimSize = std::stoul(line.c_str());

        printf("Read W = %ld\n", this->DimSize);

        std::getline(inData, line); // Skip line with NNZ
        printf("Read NNZ = %s\n", line.c_str());

        previousDocId = 0;
        firstPoint = true;
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        // Allocate memory and initialise all elements to zero.
        result.resize(this->DimSize, 0.0);

        std::string line;
        if (previousLine.empty()) {
            std::getline(inData, line);
        } else {
            line = previousLine;
        }

        do
        {
            std::vector<std::string> splits;
            boost::split(splits, line, boost::is_any_of(" "));

            auto docId = std::stoul(splits[0]);
            auto wordId = std::stoul(splits[1]) - 1; // Convert to zero-based array indexing
            auto count = static_cast<double>(std::stoul(splits[2]));

            if (firstPoint)
            {
                firstPoint = false;
                previousLine = line;
                previousDocId = docId;
            }

            if (previousDocId != docId)
            {
                // Current line belongs to the next point. Store it for later.
                previousLine = line;
                previousDocId = docId;
                break;
            }

            result[wordId] = count;

            // Read next line
            std::getline(inData, line);

        } while (inData.good());
    }
};

class EnronExperiment : public BagOfWordsExperiment
{
public:
    EnronExperiment()
    {
        
    }
};

class NYTimesExperiment : public BagOfWordsExperiment
{
public:
    NYTimesExperiment()
    {
        
    }
};


class TowerExperiment : public Experiment
{
public:
    TowerExperiment()
    {
        this->DimSize = 3UL;
        this->DataSize = 4915200UL;
        this->LowDimSize = 3UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Tower.\n");
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        result.resize(this->DimSize);

        std::string line;
        for (size_t i = 0; i < this->DimSize; i++)
        {
            std::getline(inData, line);
            result[i] = static_cast<double>(std::stol(line));
        }
    }
};

class CsvDatasetExperiment : public Experiment
{
    size_t lineNo;
public:
    CsvDatasetExperiment() : lineNo(0)
    {
        this->LowDimSize = 50L;
    }

    void parsePoint(std::vector<double> &result, std::istream &inData)
    {
        namespace io = boost::iostreams;
        namespace x3 = boost::spirit::x3;

        std::string line;
        std::getline(inData, line);
        lineNo++;

        if (line.size() == 0)
        {
            std::cout << "Skipping line " << lineNo << " because it is empty.\n";
            return;
        }

        if (!x3::phrase_parse(line.begin(), line.end(), (x3::double_ % ','), x3::space, result))
        {
            std::cout << "Failed to parse line " << lineNo << ": \n <" << line << ">\n";
        }
    }
};

class HardInstanceExperiment : public CsvDatasetExperiment
{
public:
    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing a Hard Instance dataset.\n");

        const std::regex rexp("-k(\\d+)-alpha(\\d+)");
        std::smatch matches;
        size_t k = 0, alpha = 0;

        if (std::regex_search(InputFilePath, matches, rexp) && matches.size() == 3)
        {
            k = std::stol(matches[1].str());
            alpha = std::stol(matches[2].str());

            this->DimSize = alpha * k;
            this->DataSize = std::pow(k, alpha);

            printf("Extracted\n - k=%ld\n - alpha=%ld\n - N=%ld\n - D=%ld\n"  , k, alpha, DataSize, DimSize);
        }
        else
        {
            std::cout << "Cannot extract k and alpha from file path: " << InputFilePath << "\n";
            throw std::invalid_argument("Invalid file path");
        }
    }
};

class CovertypeLowDExperiment : public CsvDatasetExperiment
{
public:
    CovertypeLowDExperiment()
    {
        this->DimSize = 54UL;
        this->DataSize = 581012UL;
        this->LowDimSize = 50UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Low-D Covertype .\n");
    }
};

class CensusLowDExperiment : public CsvDatasetExperiment
{
public:
    CensusLowDExperiment()
    {
        this->DimSize = 68UL;
        this->DataSize = 2458285UL;
        this->LowDimSize = 50UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Low-D Census .\n");
    }
};

class Caltech101Experiment : public CsvDatasetExperiment
{
public:
    Caltech101Experiment()
    {
        this->DimSize = 128UL;
        this->DataSize = 3680458UL;
        this->LowDimSize = 50UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Caltech101 dataset.\n");
    }
};

class Caltech101LowDExperiment : public CsvDatasetExperiment
{
public:
    Caltech101LowDExperiment()
    {
        this->DimSize = 128UL;
        this->DataSize = 3680458UL;
        this->LowDimSize = 50UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing Low-D Caltech101 dataset.\n");
    }
};

class NYTimes100DExperiment : public CsvDatasetExperiment
{
public:
    NYTimes100DExperiment()
    {
        this->DimSize = 100UL;
        this->DataSize = 300000UL;
        this->LowDimSize = 50UL;
    }

    void prepareFileStream(std::istream &inData)
    {
        printf("Preparing NYTimes (100-d) dataset.\n");
    }
};


class NYTimesPcaLowDExperiment : public CsvDatasetExperiment
{
public:
    NYTimesPcaLowDExperiment(const size_t dimSize)
    {
        this->DimSize = dimSize;
        this->DataSize = 300000UL;
        this->LowDimSize = dimSize;
    }

    void prepareFileStream(std::istream &inData)
    {
        std::cout << "Preparing NYTimes+PCA (d=" << DimSize << ") dataset." << std::endl;
    }
};

#endif
