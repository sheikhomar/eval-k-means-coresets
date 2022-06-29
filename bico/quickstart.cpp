#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <ctime>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <map>

#include <boost/algorithm/string.hpp>

#include "src/point/l2metric.h"
#include "src/point/squaredl2metric.h"
#include "src/point/point.h"
#include "src/point/pointweightmodifier.h"
#include "src/clustering/bico.h"
#include "src/misc/randomness.h"
#include "src/misc/randomgenerator.h"
#include "src/misc/stopwatch.h"
#include "src/datastructure/proxysolution.h"
#include "src/point/pointcentroid.h"
#include "src/point/pointweightmodifier.h"
#include "src/point/realspaceprovider.h"
#include "src/experiments/experiment.h"

using namespace CluE;
using namespace std::chrono;

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::cout << "Usage: dataset k T output_path [seed]" << std::endl;
        std::cout << "  dataset     = dataset name" << std::endl;
        std::cout << "  data_path   = file path to dataset" << std::endl;
        std::cout << "  k           = number of desired centers" << std::endl;
        std::cout << "  m           = coreset size" << std::endl;
        std::cout << "  seed        = random seed" << std::endl;
        std::cout << "  output_dir  = path to output results" << std::endl;
        std::cout << std::endl;
        std::cout << "6 arguments expected, got " << argc - 1 << ":" << std::endl;
        for (int i = 1; i < argc; ++i)
            std::cout << " " << i << ": " << argv[i] << std::endl;
        return 1;
    }

    std::string datasetName(argv[1]);
    std::string dataFilePath(argv[2]);
    size_t k = std::stoul(argv[3]);
    size_t m = std::stoul(argv[4]);
    int randomSeed = std::stoi(argv[5]);
    std::string outputDir(argv[6]);

    boost::algorithm::to_lower(datasetName);
    boost::algorithm::trim(datasetName);

    std::cout << "Running BICO with following parameters:\n";
    std::cout << " - Dataset:      " << datasetName << "\n";
    std::cout << " - Input path:   " << dataFilePath << "\n";
    std::cout << " - Clusters:     " << k << "\n";
    std::cout << " - Coreset size: " << m << "\n";
    std::cout << " - Random Seed:  " << randomSeed << "\n";
    std::cout << " - Output dir:   " << outputDir << "\n";

    if (randomSeed != -1)
    {
        std::cout << "Initializing randomess with random seed: " << randomSeed << "\n";
        Randomness::initialize(randomSeed);
    }

    std::shared_ptr<Experiment> experiment;
    if (datasetName == "census")
    {
        experiment = std::make_shared<CensusExperiment>();
    }
    else if (datasetName == "covertype")
    {
        experiment = std::make_shared<CovertypeExperiment>();
    }
    else if (datasetName == "enron")
    {
        experiment = std::make_shared<EnronExperiment>();
    }
    else if (datasetName == "tower")
    {
        experiment = std::make_shared<TowerExperiment>();
    }
    else if (datasetName.find("hardinstance") != std::string::npos)
    {
        experiment = std::make_shared<HardInstanceExperiment>();
    }
    else if (datasetName == "censuslowd")
    {
        experiment = std::make_shared<CensusLowDExperiment>();
    }
    else if (datasetName == "covertypelowd")
    {
        experiment = std::make_shared<CovertypeLowDExperiment>();
    }
    else if (datasetName == "caltech101")
    {
        experiment = std::make_shared<Caltech101Experiment>();
    }
    else if (datasetName == "caltech101lowd")
    {
        experiment = std::make_shared<Caltech101LowDExperiment>();
    }
    else if (datasetName == "nytimes100d")
    {
        experiment = std::make_shared<NYTimes100DExperiment>();
    }
    else if (datasetName == "nytimes")
    {
        experiment = std::make_shared<NYTimesExperiment>();
    }
    else if (datasetName == "nytimespcalowd")
    {
        experiment = std::make_shared<NYTimesPcaLowDExperiment>(k);
    }
    else
    {
        std::cout << "Unknown dataset: " << datasetName << "\n";
    }

    experiment->InputFilePath = dataFilePath;
    experiment->ClusterSize = k;
    experiment->TargetCoresetSize = m;
    experiment->OutputDir = outputDir;
    experiment->run();
}
