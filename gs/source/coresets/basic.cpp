#include <coresets/basic.hpp>

using namespace coresets;

BasicClustering::BasicClustering(size_t targetSamplesInCoreset) : TargetSamplesInCoreset(targetSamplesInCoreset)
{
}

std::shared_ptr<Coreset>
BasicClustering::run(const blaze::DynamicMatrix<double> &data)
{
    auto coreset = std::make_shared<Coreset>(TargetSamplesInCoreset);

    // Run k-Means++ where k=T where T is the target number of points to be included in the coreset.
    clustering::KMeans kMeansAlg(TargetSamplesInCoreset, true, false, 0);

    auto result = kMeansAlg.run(data);

    auto clusterAssignments = result->getClusterAssignments();

    for (size_t c = 0; c < clusterAssignments.getNumberOfClusters(); c++)
    {
        size_t nPointsInCluster = clusterAssignments.countPointsInCluster(c);
        double weight = static_cast<double>(nPointsInCluster);
        auto center = clusterAssignments.calcCenter(data, c);
        coreset->addCenter(c, center, weight);
    }

    return coreset;
}
