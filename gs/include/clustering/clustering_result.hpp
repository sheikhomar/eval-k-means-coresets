#pragma once

#include <string>
#include <iostream>
#include <random>

#include <blaze/Math.h>
#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>

#include <clustering/cluster_assignment_list.hpp>

namespace clustering
{
    /**
     * @brief Represents the output of a clustering algorithm.
     */
    class ClusteringResult
    {
    public:
        /**
         * @brief Creates a new instance of ClusteringResult.
         * @param clusterAssignments Cluster assignments.
         * @param centroids The final centroids.
         */
        ClusteringResult(const ClusterAssignmentList &clusterAssignments, blaze::DynamicMatrix<double> &centroids);

        ClusterAssignmentList &getClusterAssignments();

        blaze::DynamicMatrix<double> &getCentroids();

    private:
        ClusterAssignmentList clusterAssignments;
        blaze::DynamicMatrix<double> centroids;
    };
}
