#include <clustering/local_search.hpp>

using namespace clustering;

class CandidateCentersGenerator
{
private:
    std::vector<size_t> points;
    size_t numOfCenters;

public:
    CandidateCentersGenerator(size_t n, size_t k) : points(n), numOfCenters(k)
    {
        init();
    }

    void init()
    {
        std::iota(points.begin(), points.end(), 0);
    }

    bool next(std::vector<size_t> &candidates)
    {
        assert(candidates.size() == numOfCenters);

        for (size_t i = 0; i < numOfCenters; i++)
        {
            candidates[i] = points[i];
        }
        std::reverse(points.begin() + static_cast<long int>(numOfCenters), points.end());
        return std::next_permutation(points.begin(), points.end());
    }
};


LocalSearch::LocalSearch(uint k, uint s) : numOfClusters(k), swapSize(s)
{
}

std::shared_ptr<ClusteringResult>
LocalSearch::run(const blaze::DynamicMatrix<double> &data)
{
    size_t n = data.rows();
    size_t k = this->numOfClusters;

    // Initialise centers using  k-Means++.
    KMeans kMeansAlg(k);
    auto clusters = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data);
    auto initialCenters = *clusters->getClusterIndices();
    auto centers = kMeansAlg.copyRows(data, initialCenters);

    // Assign points to clusters using initial centers generated by k-Means++ initialisation.
    ClusterAssignmentList clusterAssignments(n, this->numOfClusters);
    clusterAssignments.assignAll(data, centers);

    // Let the cost of the above clusterings be the best cost seen so far.
    double bestCost = clusterAssignments.getTotalCost();
    auto bestCenters = centers;
    auto swapClusterAssignments = clusterAssignments;
    auto bestClusterAssignments = swapClusterAssignments;

    for (size_t c = 0; c < k; c++)
    {
        for (size_t p = 0; p < n; p++)
        {
            // Swap one center (c) with a point (p)
            blaze::row(centers, c) = blaze::row(data, p);

            // Reassign points to potentially new centers after the swap.
            swapClusterAssignments.assignAll(data, centers);

            // The cost after the swap.
            double cost = swapClusterAssignments.getTotalCost();

            if (cost < bestCost)
            {
                bestCost = cost;
                bestCenters = centers;
                bestClusterAssignments = swapClusterAssignments;
            }
        }
    }

    return std::make_shared<ClusteringResult>(bestClusterAssignments, bestCenters);
}

std::shared_ptr<ClusteringResult>
LocalSearch::runPlusPlus(const blaze::DynamicMatrix<double> &data, size_t nSamples, size_t nIterations)
{
    utils::Random random;
    size_t n = data.rows();
    size_t k = this->numOfClusters;

    // Initialise centers using  k-Means++.
    KMeans kMeansAlg(k);
    auto clusters = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data);
    auto initialCenters = *clusters->getClusterIndices();
    auto centers = kMeansAlg.copyRows(data, initialCenters);

    // Assign points to clusters using initial centers generated by k-Means++ initialisation.
    ClusterAssignmentList clusterAssignments(n, this->numOfClusters);
    clusterAssignments.assignAll(data, centers);

    // Let the cost of the above clusterings be the best cost seen so far.
    auto bestCost = clusterAssignments.getTotalCost();
    auto bestCenters = centers;
    auto bestClusterAssignments = clusterAssignments;

    blaze::DynamicVector<size_t> bestPointsUsedAsCenters(k);
    blaze::DynamicVector<size_t> pointsUsedAsCenters(k);
    size_t swapCount = 0;

resetIteration:
    for (size_t iteration = 0; iteration < nIterations; iteration++)
    {
        auto costs = clusterAssignments.getCentroidDistances();
        auto sampledPoints = random.choice(nSamples, costs); // TODO: Without replacement?

        for (size_t c = 0; c < k; c++)
        {
            for (auto &&p : *sampledPoints)
            {
                // Swap one center (c) with a point (p)
                blaze::row(centers, c) = blaze::row(data, p);

                swapCount++;

                pointsUsedAsCenters[c] = p;

                // Reassign points to potentially new centers after the swap.
                clusterAssignments.assignAll(data, centers);

                // The cost after the swap.
                double cost = clusterAssignments.getTotalCost();

                // printf("Swaping cluster %3ld with point %3ld costs %0.5f\n", c, p, cost);

                if (cost < bestCost)
                {
                    bestPointsUsedAsCenters = pointsUsedAsCenters;

                    bestCost = cost;
                    bestCenters = centers;
                    bestClusterAssignments = clusterAssignments;
                    printf("Found new best cost: %0.5f - number of swaps performed %ld  - ", bestCost, swapCount);
                    goto resetIteration;
                }
            }
        }
    }

    return std::make_shared<ClusteringResult>(bestClusterAssignments, bestCenters);
}
