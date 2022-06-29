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
#include <utils/random.hpp>
#include <utils/stop_watch.hpp>
#include <utils/distances.hpp>

namespace clustering
{
    /**
     * @brief Implementation of the k-Means clustering algorithm.
     */
    class KMeans
    {
    public:
        /**
         * @brief Creates a new instance of KMeans.
         * @param numOfClusters The number of clusters to generate.
         * @param initKMeansPlusPlus Initialise centroids using k-Means++.
         * @param maxIterations Maximum number of iterations.
         * @param convergenceDiff The difference in the norms of the centroids when to stop k-Means iteration.
         */
        KMeans(size_t numOfClusters, bool initKMeansPlusPlus = true, size_t maxIterations = 300, double convergenceDiff = 0.0001);

        /**
         * @brief Runs the algorithm.
         * @param data A NxD data matrix containing N data points where each point has D dimensions.
         */
        std::shared_ptr<ClusteringResult>
        run(const blaze::DynamicMatrix<double> &data);

        /**
         * @brief Picks `k` points as the initial centers using the k-Means++ initialisation procedure.
         * @param dataMatrix A NxD data matrix containing N data points where each point has D dimensions.
         */
        std::shared_ptr<clustering::ClusterAssignmentList>
        pickInitialCentersViaKMeansPlusPlus(const blaze::DynamicMatrix<double> &dataMatrix);

        blaze::DynamicMatrix<double>
        copyRows(const blaze::DynamicMatrix<double> &data, const std::vector<size_t> &indicesToCopy);

    private:
        const size_t NumOfClusters;
        const bool InitKMeansPlusPlus;
        const size_t MaxIterations;
        const double ConvergenceDiff;

        /**
         * @brief Run Lloyd's algorithm to perform the clustering of data points.
         * @param dataMatrix A NxD data matrix containing N data points where each point has D dimensions.
         * @param dataMatrix Initial k centroids where k is the number of required clusters.
         */
        std::shared_ptr<ClusteringResult>
        runLloydsAlgorithm(const blaze::DynamicMatrix<double> &dataMatrix, blaze::DynamicMatrix<double> initialCentroids);
    };

}
