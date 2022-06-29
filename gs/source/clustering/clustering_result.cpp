#include <clustering/clustering_result.hpp>

using namespace clustering;

ClusteringResult::ClusteringResult(const ClusterAssignmentList &assignments, blaze::DynamicMatrix<double> &finalCentroids) :
    clusterAssignments(assignments), centroids(finalCentroids)
{
}

ClusterAssignmentList&
ClusteringResult::getClusterAssignments()
{
    return this->clusterAssignments;
}

blaze::DynamicMatrix<double>&
ClusteringResult::getCentroids()
{
    return this->centroids;
}
