#include <coresets/stream_km.hpp>

using namespace coresets;

StreamKMeans::StreamKMeans(size_t targetSamplesInCoreset) : TargetSamplesInCoreset(targetSamplesInCoreset)
{
}

std::shared_ptr<Coreset>
StreamKMeans::run(const blaze::DynamicMatrix<double> &data)
{
    auto coreset = std::make_shared<Coreset>(TargetSamplesInCoreset);

    // Run k-Means++ where k=T where T is the number of points to be included in the coreset
    clustering::KMeans kMeansAlg(TargetSamplesInCoreset);
    auto clusters = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data);

    auto clusterIndicies = *clusters->getClusterIndices();
    for (auto &&clusterIndex : clusterIndicies)
    {
        size_t nPointsInCluster = clusters->countPointsInCluster(clusterIndex);
        double weight = static_cast<double>(nPointsInCluster);
        coreset->addPoint(clusterIndex, weight);
    }

    return coreset;
}
