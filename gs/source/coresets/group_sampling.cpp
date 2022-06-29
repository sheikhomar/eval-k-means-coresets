#include <coresets/group_sampling.hpp>

using namespace coresets;

GroupSampling::GroupSampling(size_t numberOfClusters, size_t targetSamplesInCoreset, size_t beta, size_t groupRangeSize, size_t minimumGroupSamplingSize) : 
    MinimumGroupSamplingSize(minimumGroupSamplingSize),
    Beta(beta),
    TargetSamplesInCoreset(targetSamplesInCoreset),
    NumberOfClusters(numberOfClusters),
    GroupRangeSize(groupRangeSize)
{
}

std::shared_ptr<Coreset>
GroupSampling::run(const blaze::DynamicMatrix<double> &data)
{
    clustering::KMeans kMeansAlg(this->NumberOfClusters, true, 0);
    auto clusters = kMeansAlg.run(data);

    auto coreset = std::make_shared<Coreset>(this->TargetSamplesInCoreset);

    auto clusterAssignments = clusters->getClusterAssignments();

    auto rings = this->makeRings(clusterAssignments);

    auto groups = std::make_shared<GroupSet>(this->GroupRangeSize);

    groupRingPoints(clusterAssignments, rings, groups);

    groupOvershotPoints(clusterAssignments, rings, groups);

    addShortfallPointsToCoreset(data, clusterAssignments, rings, coreset);

    addSampledPointsFromGroupsToCoreset(data, clusterAssignments, groups, coreset);

    return coreset;
}


std::shared_ptr<RingSet>
GroupSampling::makeRings(const clustering::ClusterAssignmentList &clusterAssignments)
{
    // printf("Making rings...\n");

    const int ringRangeStart = -static_cast<int>(floor(std::log10(static_cast<double>(Beta))));
    const int ringRangeEnd = -ringRangeStart;
    const auto n = clusterAssignments.getNumberOfPoints();
    const auto k = clusterAssignments.getNumberOfClusters();

    auto rings = std::make_shared<RingSet>(ringRangeStart, ringRangeEnd, k);

    // Step 2: Compute the average cost for each cluster.
    auto averageClusterCosts = clusterAssignments.calcAverageClusterCosts();

    for (size_t p = 0; p < n; p++)
    {
        // The cluster index of point `p`
        size_t c = clusterAssignments.getCluster(p);

        // The cost of point `p`: cost(p, A)
        double costOfPoint = clusterAssignments.getPointCost(p);

        // The average cost of cluster `c`: Δ_c
        double averageClusterCost = (*averageClusterCosts)[c];

        bool pointPutInRing = false;
        for (int l = ringRangeStart; l <= ringRangeEnd; l++)
        {
            auto ring = rings->findOrCreate(c, l, averageClusterCost);

            // Add point if cost(p, A) is within bounds i.e. between Δ_c*2^l and Δ_c*2^(l+1)
            if (ring->tryAddPoint(p, costOfPoint))
            {
                pointPutInRing = true;

                // Since a point cannot belong to multiple rings, there is no need to
                // test whether the point `p` falls within the ring of the next range l+1.
                break;
            }
        }

        if (pointPutInRing == false)
        {
            double innerMostRingCost = averageClusterCost * std::pow(2, ringRangeStart);
            double outerMostRingCost = averageClusterCost * std::pow(2, ringRangeEnd + 1);

            if (costOfPoint < innerMostRingCost)
            {
                // Track shortfall points: below l's lower range i.e. l<log⁡(1/β)
                rings->addShortfallPoint(p, c, costOfPoint, innerMostRingCost);
            }
            else if (costOfPoint > outerMostRingCost)
            {
                // Track overshot points: above l's upper range i.e., l>log⁡(β)
                rings->addOvershotPoint(p, c, costOfPoint, innerMostRingCost);
            }
            else
            {
                throw std::logic_error("Point should either belong to a ring or be ringless.");
            }
        }
    }

    return rings;
}

void GroupSampling::addShortfallPointsToCoreset(const blaze::DynamicMatrix<double> &data, const clustering::ClusterAssignmentList &clusters, const std::shared_ptr<RingSet> rings, std::shared_ptr<Coreset> coresetContainer)
{
    // printf("\n\nAdding shortfall points to the coreset..\n");

    // Handle points whose costs are below the lowest ring range i.e. l < log(1/beta).
    // These are called shortfall points because they fall short of being captured by the
    // inner-most ring. These points are snapped to the center of the assigned cluster by
    // adding the centers to the coreset weighted by the number of shortfall points of
    // that cluster.
    auto k = clusters.getNumberOfClusters();

    for (size_t c = 0; c < k; c++)
    {
        // The number of shortfall points for cluster `c`
        auto nShortfallPoints = rings->getNumberOfShortfallPoints(c);

        if (nShortfallPoints == 0)
        {
            // Not all clusters may have shortfall points so skip those.
            continue;
        }

        // The weight of the coreset point for the center of cluster `c`
        double weight = static_cast<double>(nShortfallPoints);

        auto center = clusters.calcCenter(data, c);

        // Add center to the coreset.
        coresetContainer->addCenter(c, center, weight);
    }
}

void GroupSampling::groupOvershotPoints(const clustering::ClusterAssignmentList &clusters, const std::shared_ptr<RingSet> rings, std::shared_ptr<GroupSet> groups)
{
    size_t numberOfGroups = 5;
    auto k = clusters.getNumberOfClusters();
    double kDouble = static_cast<double>(k);
    double totalCost = rings->computeCostOfOvershotPoints();

    // printf("\n\nGrouping overshot points, cost(O) = %0.5f\n", totalCost);

    for (size_t c = 0; c < k; c++)
    {
        double clusterCost = rings->computeCostOfOvershotPoints(c);
        auto points = rings->getOvershotPoints(c);

        #ifdef VERBOSE_DEBUG
        printf("    Cluster i=%ld  - cost(C_i ⋂ O) = %0.4f     |C_i ⋂ O| = %ld\n", c, clusterCost, points.size());
        #endif

        if (points.size() == 0)
        {
            // If no overshot points in the current cluster then go to next cluster.
            continue;
        }

        for (size_t j = 0; j < numberOfGroups; j++)
        {
            double jDouble = static_cast<double>(j);
            double lowerBound = 1 / kDouble * pow(2, -jDouble) * totalCost;
            double upperBound = 1 / kDouble * pow(2, -jDouble + 1) * totalCost;
            bool shouldAddPointsIntoGroup = false;

            if (j == 0)
            {
                // Group 0 has no upper bound. Notice this can be written as two-liners,
                // but is expanded to make it easier to read the code.
                shouldAddPointsIntoGroup = clusterCost >= lowerBound;

                #ifdef VERBOSE_DEBUG
                printf("\n      Group j=%ld    lowerBoundCost=%0.4f\n", j, lowerBound);
                #endif
            }
            else if (j == groups->GroupRangeSize-1)
            {
                // Group H has no lower bound.
                // TODO: Not part of the original algorithm as described by Chris. Check with Chris.
                shouldAddPointsIntoGroup = clusterCost < upperBound;
                
                #ifdef VERBOSE_DEBUG
                printf("\n      Group j=%ld       upperBoundCost=%0.4f\n", j, upperBound);
                #endif
            }
            else
            {
                shouldAddPointsIntoGroup = clusterCost >= lowerBound && clusterCost < upperBound;

                #ifdef VERBOSE_DEBUG
                printf("\n      Group j=%ld    lowerBoundCost=%0.4f   upperBoundCost=%0.4f\n", j, lowerBound, upperBound);
                #endif
            }

            if (shouldAddPointsIntoGroup)
            {
                auto l = std::numeric_limits<int>().max();
                auto group = groups->create(j, l, lowerBound, upperBound);

                #ifdef VERBOSE_DEBUG
                printf("            Adding %ld points to G[l=%d, j=%ld]\n", points.size(), l, j);
                #endif

                for (size_t i = 0; i < points.size(); i++)
                {
                    auto point = points[i];
                    group->addPoint(point->PointIndex, point->ClusterIndex, point->PointCost);
                }
            }
        }
    }
}

void GroupSampling::groupRingPoints(const clustering::ClusterAssignmentList &clusters, const std::shared_ptr<RingSet> rings, std::shared_ptr<GroupSet> groups)
{
    #ifdef VERBOSE_DEBUG
    printf("Grouping ring points...\n");
    #endif

    auto k = clusters.getNumberOfClusters();
    auto kDouble = static_cast<double>(k);
    for (int l = rings->RangeStart; l <= rings->RangeEnd; l++)
    {
        double ringCost = rings->calcRingCost(l);
        size_t nRingPointsForAllClusters = rings->countRingPoints(l);
        size_t nGroupedPoints = 0;

        #ifdef VERBOSE_DEBUG
        printf("\n\nRing l=%d   -  cost(R_l) = %0.4f   -   |R_l| = %ld\n", l, ringCost, nRingPointsForAllClusters);
        #endif

        for (size_t c = 0; c < k; c++)
        {
            auto ring = rings->find(c, l);
            if (ring == nullptr)
            {
                // No ring exist for the given range value `l` and cluster `c`
                continue;
            }

            auto clusterCost = ring->getTotalCost();
            auto ringPoints = ring->getPoints();
            
            #ifdef VERBOSE_DEBUG
            printf("    Cluster i=%ld  - cost(R_{l,i}) = %0.4f     |R_{l,i}| = %ld\n", c, clusterCost, ring->countPoints());
            #endif

            if (ring->countPoints() == 0)
            {
                // If nothing is captured by the current ring, then continue to the next ring.
                continue;
            }

            for (size_t j = 0; j < groups->GroupRangeSize; j++)
            {
                double jDouble = static_cast<double>(j);
                double lowerBound = 1 / kDouble * pow(2, -jDouble) * ringCost;
                double upperBound = 1 / kDouble * pow(2, -jDouble + 1) * ringCost;
                bool shouldAddPointsIntoGroup = false;

                if (j == 0)
                {
                    // Group 0 has no upper bound. Notice this can be written as two-liners,
                    // but is expanded to make it easier to read the code.
                    shouldAddPointsIntoGroup = clusterCost >= lowerBound;
                    
                    #ifdef VERBOSE_DEBUG
                    printf("\n      Group j=%ld    lowerBoundCost=%0.4f\n", j, lowerBound);
                    #endif
                }
                else if (j == groups->GroupRangeSize-1)
                {
                    // Group H has no lower bound.
                    // TODO: Not part of the original algorithm as described by Chris. Check with Chris.
                    shouldAddPointsIntoGroup = clusterCost < upperBound;
                    
                    #ifdef VERBOSE_DEBUG
                    printf("\n      Group j=%ld       upperBoundCost=%0.4f\n", j, upperBound);
                    #endif
                }
                else
                {
                    shouldAddPointsIntoGroup = clusterCost >= lowerBound && clusterCost < upperBound;
                    
                    #ifdef VERBOSE_DEBUG
                    printf("\n      Group j=%ld    lowerBoundCost=%0.4f   upperBoundCost=%0.4f\n", j, lowerBound, upperBound);
                    #endif
                }

                if (shouldAddPointsIntoGroup)
                {
                    // Points which belong to cluster `c` and ring `l`
                    #ifdef VERBOSE_DEBUG
                    printf("            Adding %ld points to G[l=%d, j=%ld]\n", ringPoints.size(), l, j);
                    #endif

                    auto group = groups->create(j, l, lowerBound, upperBound);
                    for (size_t i = 0; i < ringPoints.size(); i++)
                    {
                        auto ringPoint = ringPoints[i];
                        group->addPoint(ringPoint->PointIndex, ringPoint->ClusterIndex, ringPoint->Cost);
                        nGroupedPoints++;
                    }
                }
            }
        }

        if (nRingPointsForAllClusters != nGroupedPoints)
        {
            printf("Not all points in ring l=%d are put in a group. ", l);
            printf("Number of points in the ring is %ld but only %ld points are grouped.\n",
                   nRingPointsForAllClusters, nGroupedPoints);
        }
        assert(nRingPointsForAllClusters == nGroupedPoints);
    }
}

void GroupSampling::addSampledPointsFromGroupsToCoreset(const blaze::DynamicMatrix<double> &data, const clustering::ClusterAssignmentList &clusterAssignments, const std::shared_ptr<GroupSet> groups, std::shared_ptr<Coreset> coresetContainer)
{
    #ifdef VERBOSE_DEBUG
    printf("\n\nSampling from groups...\n");
    #endif

    const size_t minSamplingSize = 1;
    auto T = coresetContainer->TargetSize;
    auto TDouble = static_cast<double>(T);

    auto totalCost = clusterAssignments.getTotalCost();
    auto k = clusterAssignments.getNumberOfClusters();

    #ifdef VERBOSE_DEBUG
    printf("  Minimum size before sampling from any group is %ld...\n", minSamplingSize);
    printf("  cost(A) = %0.5f...\n", totalCost);
    printf("  T = %ld...\n", T);
    #endif

    // The number of remaining points needed for the coreset.
    size_t T_remaining = T;
    

    // Tracks the groups that we need to sample points from.
    std::vector<size_t> samplingGroupIndices;

    // Track the total cost of the groups that we need to sample points from.
    double samplingGroupTotalCost = 0.0;

    for (size_t m = 0; m < groups->size(); m++)
    {
        auto group = groups->at(m);
        auto groupPoints = group->getPoints();
        auto groupCost = group->calcTotalCost();
        auto normalizedGroupCost = groupCost / totalCost;
        auto numSamples = TDouble * normalizedGroupCost;

        #ifdef VERBOSE_DEBUG
        printf("\n    Group m=%ld:   |G_m|=%2ld   cost(G_m)=%2.4f   cost(G_m)/cost(A)=%0.4f   T_m=%0.5f \n",
               m, groupPoints.size(), group->calcTotalCost(), normalizedGroupCost, numSamples);
        #endif

        if (numSamples < minSamplingSize)
        {
            #ifdef VERBOSE_DEBUG
            printf("        Will not sample because T_m is below threshold...\n");
            #endif

            for (size_t c = 0; c < k; c++)
            {
                auto nPointsInCluster = group->countPointsInCluster(c);
                if (nPointsInCluster > 0)
                {
                    double weight = static_cast<double>(nPointsInCluster);
                    auto center = clusterAssignments.calcCenter(data, c);
                    coresetContainer->addCenter(c, center, weight);
                }
            }
        }
        else if (numSamples >= static_cast<double>(groupPoints.size()))
        {
            #ifdef VERBOSE_DEBUG
            printf("        Will not sample because T_m >= |G_m|.\n");
            #endif

            for (size_t i = 0; i < groupPoints.size(); i++)
            {
                coresetContainer->addPoint(groupPoints[i]->PointIndex, 1.0);
            }
            T_remaining -= groupPoints.size();
        }
        else
        {
            #ifdef VERBOSE_DEBUG
            printf("        Will sample later.\n");
            #endif

            samplingGroupIndices.push_back(m);
            samplingGroupTotalCost += groupCost;
        }
    }

    #ifdef VERBOSE_DEBUG
    // Now, we have to deal with the groups that we can sample points from.
    printf("\n\nDealing with the groups that we can sample points from...");
    #endif
    
    for (auto &m : samplingGroupIndices)
    {
        auto group = groups->at(m);
        auto groupCost = group->calcTotalCost();
        auto groupPoints = group->getPoints();
        auto normalizedGroupCost = groupCost / samplingGroupTotalCost;
        auto numSamplesReal = static_cast<double>(T_remaining) * normalizedGroupCost;
        auto numSamplesInt = random.stochasticRounding(numSamplesReal);

        #ifdef VERBOSE_DEBUG
        printf("\n    Group m=%ld:   |G_m|=%2ld   cost(G_m)=%2.4f   cost(G_m)/cost(A)=%0.4f   T_m=%0.5f  round(T_m)=%ld \n",
               m, groupPoints.size(), group->calcTotalCost(), normalizedGroupCost, numSamplesReal, numSamplesInt);
        #endif

        auto sampledPoints = random.choice(groupPoints, numSamplesInt);

        #ifdef VERBOSE_DEBUG
        printf("        Sampled points from group:\n");
        #endif
        
        for (size_t i = 0; i < sampledPoints.size(); i++)
        {
            auto sampledPoint = sampledPoints[i];
            auto weight = groupCost / (numSamplesReal * sampledPoint->Cost);
            coresetContainer->addPoint(sampledPoint->PointIndex, weight);
        }
    }
}

// void GroupSampling::printPythonCodeForVisualisation(std::shared_ptr<clustering::ClusteringResult> result, std::shared_ptr<RingSet> rings)
// {
//     auto clusterAssignments = result->getClusterAssignments();
//     auto centers = result->getCentroids();
//     auto k = clusterAssignments.getNumberOfClusters();
//     auto n = clusterAssignments.getNumberOfPoints();

//     printf("k = %ld\n", k);
//     printf("cluster_labels = [");
//     for (size_t p = 0; p < n; p++)
//     {
//         printf("%ld, ", clusterAssignments.getCluster(p));
//     }
//     printf("]\n");

//     printf("cluster_centers = np.array([\n");
//     for (size_t c = 0; c < centers.rows(); c++)
//     {
//         printf("  [");
//         for (size_t d = 0; d < centers.columns(); d++)
//         {
//             printf("%0.5f, ", centers.at(c, d));
//         }
//         printf("],\n");
//     }
//     printf("])\n");

//     printf("ring_ranges = [");
//     for (int l = rings->RangeStart; l <= rings->RangeEnd; l++)
//     {
//         printf("%d, ", l);
//     }
//     printf("]\n");

//     printf("rings = np.array([\n");
//     for (size_t c = 0; c < k; c++)
//     {
//         printf("  [");
//         for (int l = rings->RangeStart; l <= rings->RangeEnd; l++)
//         {
//             auto ring = rings->find(c, l);
//             printf("%0.5f, ", ring->getLowerBoundCost());
//         }
//         printf("],\n");
//     }
//     printf("])\n");

//     printf("rings_upper_bounds = np.array([\n");
//     for (size_t c = 0; c < k; c++)
//     {
//         printf("  [");
//         for (int l = rings->RangeStart; l <= rings->RangeEnd; l++)
//         {
//             auto ring = rings->find(c, l);
//             printf("%0.5f, ", ring->getUpperBoundCost());
//         }
//         printf("],\n");
//     }
//     printf("])\n");
// }
