#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <clustering/clustering_result.hpp>
#include <clustering/kmeans.hpp>
#include <coresets/coreset.hpp>
#include <utils/random.hpp>

namespace coresets
{
    /**
     * Represents a reference to a point which has been assigned a cluster.
     */
    struct ClusteredPoint
    {
        /**
         * The index of the point in the dataset.
         */
        const size_t PointIndex;

        /**
         * The index of the cluster for which this point is assigned.
         */
        const size_t ClusterIndex;

        /**
         * The cost of this point in its assigned cluster. 
         * 
         * The cost is the distance to the assigned cluster's center.
         */
        const double Cost;

        ClusteredPoint(size_t pointIndex, size_t clusterIndex, double cost) : PointIndex(pointIndex), ClusterIndex(clusterIndex), Cost(cost)
        {
        }

        ClusteredPoint &operator=(const ClusteredPoint &) = delete; // Disallow assignment
    };

    /**
     * Represents a group which is uniquely identified by its range value (j) and its ring range value (l) i.e., G_{j,l}
     */
    class Group
    {

    public:
        /**
         * The group's range value `j`, it is a non-negative value.
         */
        const size_t RangeValue;

        /**
         * The group's ring range value i.e., `l`.
         */
        const int RingRangeValue;

        /**
         * The lower bound cost of the group.
         */
        const double LowerBoundCost;

        /**
         * The upper bound cost of the group.
         */
        const double UpperBoundCost;

        Group(size_t rangeValue, int ringRangeValue, double lowerBoundCost, double upperBoundCost) : RangeValue(rangeValue), RingRangeValue(ringRangeValue), LowerBoundCost(lowerBoundCost), UpperBoundCost(upperBoundCost)
        {
        }

        Group &operator=(const Group &) = delete; // Disallow assignment

        void addPoint(size_t point, size_t cluster, double cost)
        {
            points.push_back(std::make_shared<ClusteredPoint>(point, cluster, cost));
        }

        const std::vector<std::shared_ptr<ClusteredPoint>> &
        getPoints() const
        {
            return points;
        }

        double
        calcTotalCost() const
        {
            double sum = 0;
            for (size_t i = 0; i < points.size(); i++)
            {
                sum += points[i]->Cost;
            }

            return sum;
        }

        /**
         * @brief Counts the number of points in the given cluster.
         */
        size_t
        countPointsInCluster(size_t clusterIndex) const
        {
            size_t count = 0;
            for (size_t i = 0; i < points.size(); i++)
            {
                if (points[i]->ClusterIndex == clusterIndex)
                {
                    count++;
                }
            }
            return count;
        }

    private:
        /**
         * The points assigned to this group.
         */
        std::vector<std::shared_ptr<ClusteredPoint>> points;
    };

    class GroupSet
    {
        std::vector<std::shared_ptr<Group>> groups;

    public:
        const size_t GroupRangeSize;

        GroupSet(size_t groupRangeSize) : GroupRangeSize(groupRangeSize)
        {
        }

        GroupSet &operator=(const GroupSet &) = delete; // Disallow assignment

        std::shared_ptr<Group> create(size_t rangeValue, int ringRangeValue, double lowerBoundCost, double upperBoundCost)
        {
            auto group = std::make_shared<Group>(rangeValue, ringRangeValue, lowerBoundCost, upperBoundCost);
            groups.push_back(group);
            return group;
        }

        size_t size() const
        {
            return this->groups.size();
        }

        std::shared_ptr<Group>
        operator[](size_t index) const
        {
            return this->groups[index];
        }

        std::shared_ptr<Group>
        at(size_t index) const
        {
            return this->groups[index];
        }

        blaze::DynamicVector<double>
        calcNormalizedCosts() const
        {
            blaze::DynamicVector<double> costs(this->groups.size());
            double sumOfGroupCosts = 0.0;
            for (size_t i = 0; i < this->groups.size(); i++)
            {
                auto groupCost = this->groups[i]->calcTotalCost();
                costs[i] = groupCost;
                sumOfGroupCosts += groupCost;
            }

            return costs / sumOfGroupCosts;
        }
    };

    /**
     * Represents a point that is not captured by any ring. 
     */
    struct RinglessPoint
    {
        const size_t PointIndex;
        const size_t ClusterIndex;
        const double PointCost;
        const double CostBoundary;
        const bool IsOvershot;

        RinglessPoint(size_t postIndex, size_t clusterIndex, double pointCost, double costBoundary, bool isOvershot) : PointIndex(postIndex), ClusterIndex(clusterIndex), PointCost(pointCost), CostBoundary(costBoundary), IsOvershot(isOvershot)
        {
        }

        RinglessPoint &operator=(const RinglessPoint &) = delete; // Disallow assignment
    };

    class Ring
    {
    public:
        /**
         * The cluster for which this ring belongs to.
         */
        const size_t ClusterIndex;

        /**
         * The range value of this ring i.e., `l`.
         */
        const int RangeValue;

        /**
         * The average cost for the cluster associated with this ring.
         */
        const double AverageClusterCost;

        Ring(size_t clusterIndex, int rangeValue, double averageClusterCost) : ClusterIndex(clusterIndex), RangeValue(rangeValue), AverageClusterCost(averageClusterCost)
        {
            // Ring upper bound cost := Δ_c * 2^l
            LowerBoundCost = averageClusterCost * std::pow(2, rangeValue);

            // Ring upper bound cost := Δ_c * 2^(l+1)
            UpperBoundCost = averageClusterCost * std::pow(2, rangeValue + 1);

            TotalCost = 0.0;
        }

        Ring &operator=(const Ring &) = delete; // Disallow assignment

        /**
         * @brief Adds a point to the ring if its costs is within the bounds of this ring.
         * @param pointIndex The index of the point to add to this ring.
         * @param cost The cost of the point i.e., cost(p, A)
         * @return `true` if points is added to the ring, otherwise `false`
         */
        bool
        tryAddPoint(size_t pointIndex, double cost)
        {
            if (isCostWithinBounds(cost))
            {
                #ifdef VERBOSE_DEBUG
                printf("Ring Point %3ld with cost(p, A) = %0.4f  ->  R[%2d, %ld]  [%0.4f, %0.4f) \n",
                       pointIndex, cost, RangeValue, ClusterIndex, LowerBoundCost, UpperBoundCost);
                #endif

                points.push_back(std::make_shared<ClusteredPoint>(pointIndex, ClusterIndex, cost));
                TotalCost += cost;
                return true;
            }

            return false;
        }

        bool
        isCostWithinBounds(double cost)
        {
            // If cost(p, A) is between Δ_c*2^l and Δ_c*2^(l+1) ...
            return cost >= LowerBoundCost && cost < UpperBoundCost;
        }

        double getLowerBoundCost() { return LowerBoundCost; }
        double getUpperBoundCost() { return UpperBoundCost; }

        /**
         * @brief Sums the costs of all points in captured by this ring.
         */
        double getTotalCost() { return TotalCost; }

        const std::vector<std::shared_ptr<ClusteredPoint>> &
        getPoints() const
        {
            return this->points;
        }

        size_t
        countPoints() const
        {
            return this->points.size();
        }

    private:
        /**
         * The points assigned to this ring.
         */
        std::vector<std::shared_ptr<ClusteredPoint>> points;

        double LowerBoundCost;

        double UpperBoundCost;

        /**
         * The sum of the point cost in this ring.
         */
        double TotalCost;
    };

    class RingSet
    {
        std::vector<std::shared_ptr<Ring>> rings;
        std::vector<std::shared_ptr<RinglessPoint>> overshotPoints;
        std::vector<std::shared_ptr<RinglessPoint>> shortfallPoints;

    public:
        const int RangeStart;
        const int RangeEnd;
        const size_t NumberOfClusters;

        RingSet(int start, int end, size_t numberOfClusters) : RangeStart(start), RangeEnd(end), NumberOfClusters(numberOfClusters)
        {
        }

        RingSet &operator=(const RingSet &) = delete; // Disallow assignment

        std::shared_ptr<Ring> find(size_t clusterIndex, int rangeValue) const
        {
            for (size_t i = 0; i < rings.size(); i++)
            {
                auto ring = rings[i];
                if (ring->ClusterIndex == clusterIndex && ring->RangeValue == rangeValue)
                {
                    return ring;
                }
            }
            return nullptr;
        }

        std::shared_ptr<Ring> findOrCreate(size_t clusterIndex, int rangeValue, double averageClusterCost)
        {
            auto ring = find(clusterIndex, rangeValue);
            if (ring == nullptr)
            {
                #ifdef VERBOSE_DEBUG
                printf("Ring for cluster=%ld and l=%2d not found. Creating...\n", clusterIndex, rangeValue);
                #endif
                ring = std::make_shared<Ring>(clusterIndex, rangeValue, averageClusterCost);
                rings.push_back(ring);
            }
            return ring;
        }

        void addOvershotPoint(size_t pointIndex, size_t clusterIndex, double cost, double costBoundary)
        {
            #ifdef VERBOSE_DEBUG
            printf("Overshot Point %3ld with cost(p, A) = %0.4f cluster(p)=%ld -> the cost(p, A) ",
                   pointIndex, cost, clusterIndex);
            printf("is above the cost range of outer most ring (%.4f)\n", costBoundary);
            #endif

            auto point = std::make_shared<RinglessPoint>(pointIndex, clusterIndex, cost, costBoundary, true);
            overshotPoints.push_back(point);
        }

        void addShortfallPoint(size_t pointIndex, size_t clusterIndex, double cost, double costBoundary)
        {
            #ifdef VERBOSE_DEBUG
            printf("Shortfall Point %3ld with cost(p, A) = %0.4f cluster(p)=%ld -> the cost(p, A) ",
                   pointIndex, cost, clusterIndex);
            printf("falls below the cost range of inner most ring (%.4f)\n", costBoundary);
            #endif

            auto point = std::make_shared<RinglessPoint>(pointIndex, clusterIndex, cost, costBoundary, false);
            shortfallPoints.push_back(point);
        }

        /**
         * @brief Sums the costs of all points in captured by all clusters for a given ring range i.e., cost(R_l) = sum_{p in R_l} cost(p, A)
         * @param ringRangeValue The ring range value i.e. l
         */
        double
        calcRingCost(int ringRangeValue) const
        {
            double sum = 0.0F;
            for (size_t i = 0; i < rings.size(); i++)
            {
                auto ring = rings[i];
                if (ring->RangeValue == ringRangeValue)
                {
                    sum += ring->getTotalCost();
                }
            }
            return sum;
        }

        /**
         * @brief Counts the number of points captured by a given ring range l i.e., |R_l|
         * @param ringRangeValue The ring range value i.e. l
         */
        size_t
        countRingPoints(int ringRangeValue) const
        {
            size_t count = 0;
            for (size_t i = 0; i < rings.size(); i++)
            {
                auto ring = rings[i];
                if (ring->RangeValue == ringRangeValue)
                {
                    count += ring->countPoints();
                }
            }
            return count;
        }

        /**
         * @brief Returns the number of shortfall points in a given cluster.
         * @param clusterIndex The cluster for which to search for shortfall points.
         */
        size_t
        getNumberOfShortfallPoints(size_t clusterIndex) const
        {
            size_t count = 0;
            for (size_t i = 0; i < shortfallPoints.size(); i++)
            {
                auto point = shortfallPoints[i];
                if (point->ClusterIndex == clusterIndex)
                {
                    count++;
                }
            }
            return count;
        }

        /**
         * @brief Computes the cost of overshot points.
         * @param clusterIndex If -1 then compute the costs for all points. If non-negative, computes the cost for all points in the given cluster.
         */
        double
        computeCostOfOvershotPoints(int clusterIndex = -1) const
        {
            double cost = 0.0;
            for (size_t i = 0; i < this->overshotPoints.size(); i++)
            {
                auto point = this->overshotPoints.at(i);

                if (clusterIndex == -1 || (clusterIndex >= 0 && point->ClusterIndex == static_cast<size_t>(clusterIndex)))
                {
                    cost += point->PointCost;
                }
            }

            return cost;
        }

        /**
         * @brief Computes the cost of  points in a given cluster.
         * @param clusterIndex The cluster for which to compute cost.
         */
        double
        computeCostOfOvershotPoints(size_t clusterIndex) const
        {
            return computeCostOfOvershotPoints(static_cast<int>(clusterIndex));
        }

        std::vector<std::shared_ptr<RinglessPoint>>
        getOvershotPoints(size_t clusterIndex) const
        {
            std::vector<std::shared_ptr<RinglessPoint>> filteredPoints;
            for (size_t i = 0; i < this->overshotPoints.size(); i++)
            {
                auto point = this->overshotPoints.at(i);
                if (point->ClusterIndex == clusterIndex)
                {
                    filteredPoints.push_back(point);
                }
            }
            return filteredPoints;
        }
    };

    class GroupSampling
    {
    public:
        /**
         * The minimum sampling size before attempting to sample points from a group: T_s
         */
        const size_t MinimumGroupSamplingSize;

        /**
         * Used to compute the ring ranges: ell in [-log(beta), log(beta)]
         */
        const size_t Beta;

        /**
         * Number of points that the algorithm should aim to include in the coreset: T
         */
        const size_t TargetSamplesInCoreset;

        /**
         * Number of clusters to partition the data into: k
         */
        const size_t NumberOfClusters;

        /**
         * The size of the group range: H
         */
        const size_t GroupRangeSize;

        GroupSampling(size_t numberOfClusters, size_t targetSamplesInCoreset, size_t beta, size_t groupRangeSize, size_t minimumGroupSamplingSize);

        std::shared_ptr<Coreset>
        run(const blaze::DynamicMatrix<double> &data);

    private:
        utils::Random random;

        std::shared_ptr<RingSet>
        makeRings(const clustering::ClusterAssignmentList &clusters);

        /**
         * @brief Add points inside doughnut holes i.e., points that are closest to cluster centers but are not captured by any rings.
         */
        void addShortfallPointsToCoreset(const blaze::DynamicMatrix<double> &data, const clustering::ClusterAssignmentList &clusters, const std::shared_ptr<RingSet> rings, std::shared_ptr<Coreset> coresetContainer);

        /**
         * @brief Group overshot points i.e., points that are far from cluster centers and are not captured by any rings.
         */
        void groupOvershotPoints(const clustering::ClusterAssignmentList &clusters, const std::shared_ptr<RingSet> rings, std::shared_ptr<GroupSet> groups);

        /**
         * @brief Group points arranged in rings.
         */
        void groupRingPoints(const clustering::ClusterAssignmentList &clusters, const std::shared_ptr<RingSet> rings, std::shared_ptr<GroupSet> groups);

        void addSampledPointsFromGroupsToCoreset(const blaze::DynamicMatrix<double> &data, const clustering::ClusterAssignmentList &clusterAssignments, const std::shared_ptr<GroupSet> groups, std::shared_ptr<Coreset> coresetContainer);

        void printPythonCodeForVisualisation(std::shared_ptr<clustering::ClusteringResult> result, std::shared_ptr<RingSet> rings);
    };
}
