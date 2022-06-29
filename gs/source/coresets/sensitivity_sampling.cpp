#include <coresets/sensitivity_sampling.hpp>

using namespace coresets;

SensitivitySampling::SensitivitySampling(size_t numberOfClusters, size_t targetSamplesInCoreset) : TargetSamplesInCoreset(targetSamplesInCoreset),
                                                                                                   NumberOfClusters(numberOfClusters)

{
}

std::shared_ptr<Coreset>
SensitivitySampling::run(const blaze::DynamicMatrix<double> &data)
{
    clustering::KMeans kMeansAlg(NumberOfClusters, true, 0);

    auto result = kMeansAlg.run(data);

    auto clusterAssignments = result->getClusterAssignments();

    auto coreset = generateCoresetPoints(data, clusterAssignments);

    return coreset;
}

std::shared_ptr<Coreset>
SensitivitySampling::generateCoresetPoints(const blaze::DynamicMatrix<double> &data, const clustering::ClusterAssignmentList &clusterAssignments)
{
    auto coreset = std::make_shared<Coreset>(TargetSamplesInCoreset);

    // Step 2b: compute cost(A). Assume it is the sum of all costs.
    auto sumOfCosts = clusterAssignments.getTotalCost();

    // Step 2c: compute the sampling distribution: cost(p, A)/cost(A)
    auto samplingDistribution = clusterAssignments.getNormalizedCosts();

    auto sampledIndices = random.choice(TargetSamplesInCoreset, samplingDistribution);

    double T = static_cast<double>(TargetSamplesInCoreset);

    // Loop through the sampled points and calculate
    // the weight associated with each of these points.
    for (size_t j = 0; j < TargetSamplesInCoreset; j++)
    {
        size_t sampledPointIndex = (*sampledIndices)[j];

        // We scale the cost of the sampled point by a factor of T i.e. T * cost(p,A)
        double scaledCostPofA = T * clusterAssignments.getPointCost(sampledPointIndex);

        // The weight of the sampled point is now: cost(A) / (T*cost(p,A))
        double weight = sumOfCosts / scaledCostPofA;

        coreset->addPoint(sampledPointIndex, weight);

        // printf("Sampled point %3ld gets weight %.5f \n", sampledPointIndex, weight);
    }

    auto numberOfClusters = clusterAssignments.getNumberOfClusters();
    auto centerWeights = calcCenterWeights(clusterAssignments, sampledIndices);

    for (size_t c = 0; c < numberOfClusters; c++)
    {
        auto weight = (*centerWeights)[c];
        auto center = clusterAssignments.calcCenter(data, c);
        coreset->addCenter(c, center, weight);
    }

    return coreset;
}

std::shared_ptr<blaze::DynamicVector<double>>
SensitivitySampling::calcCenterWeights(
    const clustering::ClusterAssignmentList &clusterAssignments,
    std::shared_ptr<blaze::DynamicVector<size_t>> sampledIndices)
{
    auto sumOfCosts = clusterAssignments.getTotalCost();
    auto numberOfClusters = clusterAssignments.getNumberOfClusters();

    // Initialise an array to store center weights w_i
    auto centerWeights = std::make_shared<blaze::DynamicVector<double>>(numberOfClusters);
    centerWeights->reset();

    double T = static_cast<double>(TargetSamplesInCoreset);

    // For each of the T sampled points...
    for (auto &&p : *sampledIndices)
    {
        // Find point p's assigned cluster C_i.
        size_t clusterOfPointP = clusterAssignments.getCluster(p);

        // Find cost(p, A)
        double costPOfA = clusterAssignments.getPointCost(p);

        // Compute cost(A)/(T*cost(p,A))
        double weightContributionOfP = sumOfCosts / (T * costPOfA);

        // printf("Point %3ld contributes %.5f to cluster %ld  ", p, weightContributionOfP, clusterOfPointP);

        // Sum it up: sum_{p sampled and p in C_i}   cost(A)/(T*cost(p,A))
        (*centerWeights)[clusterOfPointP] += weightContributionOfP;

        // printf("  =>  w_%ld = %.5f\n", clusterOfPointP, (*centerWeights)[clusterOfPointP]);
    }

    // For each of the k' centers, compute the center weights.
    for (size_t c = 0; c < numberOfClusters; c++)
    {
        // Find precomputed center weight: w_i
        double w_i = (*centerWeights)[c];

        // Compute |C_i|
        size_t numberOfPointsInCluster = clusterAssignments.countPointsInCluster(c);

        // Compute max(0, |C_i| - w_i)
        double centerWeight = blaze::max(0.0, static_cast<double>(numberOfPointsInCluster) - w_i);

        // Update the center weight.
        (*centerWeights)[c] = centerWeight;
    }

    return centerWeights;
}
