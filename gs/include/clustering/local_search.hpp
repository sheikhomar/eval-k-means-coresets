#pragma once

#include <memory>
#include <iostream>
#include <random>
#include <string>

#include <blaze/Math.h>
#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>

#include <clustering/cluster_assignment_list.hpp>
#include <clustering/clustering_result.hpp>
#include <clustering/kmeans.hpp>
#include <utils/random.hpp>

namespace clustering
{
    class LocalSearch
    {
    public:
        /**
         * @brief Creates a new instance of LocalSearch. 
         */
        LocalSearch(uint numOfClusters, uint swapSize);

        /**
         * @brief Runs the algorithm.
         * @param data A NxD data matrix containing N data points where each point has D dimensions.
         */
        std::shared_ptr<ClusteringResult>
        run(const blaze::DynamicMatrix<double> &data);

        /**
         * @brief Runs the faster version of the algorithm.
         * @param data A NxD data matrix containing N data points where each point has D dimensions.
         */
        std::shared_ptr<ClusteringResult>
        runPlusPlus(const blaze::DynamicMatrix<double> &data, size_t nSamples, size_t nIterations);
    
    private:
        uint numOfClusters;

        uint swapSize;
    };
}