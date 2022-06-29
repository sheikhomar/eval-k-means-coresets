#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <map>

#include <clustering/clustering_result.hpp>
#include <clustering/kmeans.hpp>
#include <clustering/kmeans1d.hpp>
#include <coresets/coreset.hpp>
#include <utils/random.hpp>
#include <utils/distances.hpp>
#include <blaze/Math.h>

namespace coresets
{
    class RandomRay
    {
    private:
        utils::Random random;

    public:
        const size_t OriginIndex;
        blaze::DynamicVector<double> Direction;
        std::vector<size_t> points;
        std::vector<double> lengths;
        std::vector<double> distances;
        double DirectionDotProduct;

        RandomRay(const size_t originIndex, const size_t dimensions) : OriginIndex(originIndex), Direction(dimensions)
        {
            random.normal(Direction);
            Direction = Direction / blaze::l2Norm(Direction);

            DirectionDotProduct = 0;
            for (size_t j = 0; j < dimensions; j++)
            {
                DirectionDotProduct += Direction[j] * Direction[j];
            }
        }

        double
        computeProjectedPointLength(const blaze::DynamicMatrix<double> &data, const size_t otherPointIndex)
        {
            const size_t d = data.columns();
            double rpDotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                // Change the origin of the other point to the point given by OriginIndex
                // because the Direction vector has its origin at OriginIndex.
                auto otherPoint_j = data.at(otherPointIndex, j) - data.at(OriginIndex, j);
                rpDotProd += otherPoint_j * Direction[j];
            }
            return rpDotProd / DirectionDotProduct;
        }

        double
        distanceToPoint(const blaze::DynamicMatrix<double> &data, const size_t otherPointIndex, const double projectedPointLength)
        {
            const size_t d = data.columns();
            double dotProd = 0.0;
            for (size_t j = 0; j < d; j++)
            {
                // Change the origin of the other point to the point given by OriginIndex
                // because the Direction vector has its origin at OriginIndex.
                auto otherPoint_j = data.at(otherPointIndex, j) - data.at(OriginIndex, j);

                auto projectedPoint_j = projectedPointLength * Direction[j];
                auto diff = projectedPoint_j - otherPoint_j;
                dotProd += diff * diff;
            }
            return std::sqrt(dotProd);
        }

        void assign(const size_t pointIndex, const double distance, const double projectedPointLength)
        {
            points.push_back(pointIndex);
            distances.push_back(distance);
            lengths.push_back(projectedPointLength);
        }

        size_t
        getNumberOfPoints() const
        {
            return points.size();
        }

        size_t
        calcNumberOfOneDimensionalClusters(size_t nClusterPoints, double targetPointsFromEachCluster) const
        {
            size_t nRayPoints = points.size();
            double rayTargetProportion = static_cast<double>(nRayPoints) / static_cast<double>(nClusterPoints);
            double n1dClusters = std::ceil(rayTargetProportion * targetPointsFromEachCluster);
            return static_cast<size_t>(n1dClusters);
        }

        std::shared_ptr<std::map<size_t, std::vector<size_t>>>
        performOneDimensionalClustering(size_t numberOfClusters) const
        {
            auto clustersAndPoints = std::make_shared<std::map<size_t, std::vector<size_t>>>();
            size_t nRayPoints = points.size();

            if (numberOfClusters > 1)
            {
                std::vector<size_t> clusterLabels(nRayPoints);
                std::vector<double> centers(numberOfClusters);
                clustering::kmeans1d::cluster(this->lengths, numberOfClusters, clusterLabels.data(), centers.data());

                for (size_t c = 0; c < numberOfClusters; c++)
                {
                    std::vector<size_t> clusterPoints;
                    clustersAndPoints->emplace(c, clusterPoints);
                }

                for (size_t p = 0; p < nRayPoints; p++)
                {
                    auto pointIndex = this->points[p];
                    auto clusterLabel = clusterLabels[p];

                    clustersAndPoints->at(clusterLabel).push_back(pointIndex);
                }
            }

            return clustersAndPoints;
        }
    };

    class RayContainer
    {
    private:
        std::map<size_t, std::vector<std::shared_ptr<RandomRay>>> clusterRays;
        const size_t Dimensions;

    public:
        RayContainer(const size_t &dimensions) : clusterRays(), Dimensions(dimensions)
        {
        }

        const std::vector<std::shared_ptr<RandomRay>> &
        createRays(const size_t &numberOfRays, const size_t &centerPoint)
        {
            std::vector<std::shared_ptr<RandomRay>> rays;
            for (size_t i = 0; i < numberOfRays; i++)
            {
                rays.push_back(std::make_shared<RandomRay>(centerPoint, Dimensions));
            }
            clusterRays.emplace(centerPoint, rays);
            return clusterRays.at(centerPoint);
        }

        const std::vector<std::shared_ptr<RandomRay>> &
        getRays(const size_t &centerPoint)
        {
            return clusterRays.at(centerPoint);
        }
    };

    class RayMaker
    {
    private:
        utils::Random random;

    public:
        /**
         * Number of points that the algorithm should aim to include in the coreset: T
         */
        const size_t TargetSamplesInCoreset;
        const size_t MaxNumberOfRaysPerCluster;
        const size_t NumberOfClusters;
        const double TargetPointsFromEachCluster;

        RayMaker(size_t k, size_t targetSamplesInCoreset, size_t maxNumberOfRaysPerCluster) : TargetSamplesInCoreset(targetSamplesInCoreset),
                                                                                              MaxNumberOfRaysPerCluster(maxNumberOfRaysPerCluster),
                                                                                              NumberOfClusters(k),
                                                                                              TargetPointsFromEachCluster(static_cast<double>(targetSamplesInCoreset) / static_cast<double>(k))
        {
        }

        std::shared_ptr<Coreset>
        run(const blaze::DynamicMatrix<double> &data)
        {
            auto coreset = std::make_shared<Coreset>(TargetSamplesInCoreset);

            const size_t k = NumberOfClusters;

            // Compute initial solution S
            clustering::KMeans kMeansAlg(k);
            auto clusters = kMeansAlg.pickInitialCentersViaKMeansPlusPlus(data);
            auto rayContainer = createRays(data, clusters);

            auto centerIndicies = *clusters->getClusterIndices();
            size_t centerCounter = 0;
            for (auto &&centerPoint : centerIndicies)
            {
                auto nClusterPoints = clusters->countPointsInCluster(centerPoint);
                auto rays = rayContainer->getRays(centerPoint);

                for (auto &&ray : rays)
                {
                    size_t n1dClusters = ray->calcNumberOfOneDimensionalClusters(nClusterPoints, TargetPointsFromEachCluster);
                    auto clusteredPoints = ray->performOneDimensionalClustering(n1dClusters);

                    for (auto &&pair : *clusteredPoints)
                    {
                        auto &pointsInCluster = pair.second;
                        if (pointsInCluster.size() > 1)
                        {
                            auto center = calcCenter(data, pointsInCluster);
                            auto weight = static_cast<double>(pointsInCluster.size());
                            coreset->addCenter(centerCounter, center, weight);
                            centerCounter++;
                        }
                    }
                }
            }

            return coreset;
        }

        std::shared_ptr<RayContainer>
        createRays(const blaze::DynamicMatrix<double> &data, std::shared_ptr<clustering::ClusterAssignmentList> clusters)
        {
            const size_t d = data.columns();
            auto rayContainer = std::make_shared<RayContainer>(d);
            auto centerIndicies = *clusters->getClusterIndices();

            for (auto &&centerPoint : centerIndicies)
            {
                auto points = clusters->getPointsByCluster(centerPoint);

                double numberOfPointInCluster = static_cast<double>(points->size());
                double numberOfRays = std::min(
                    static_cast<double>(MaxNumberOfRaysPerCluster),
                    std::ceil(numberOfPointInCluster / (TargetPointsFromEachCluster * 2)));
                size_t nRays = static_cast<size_t>(numberOfRays);
                auto clusterRays = rayContainer->createRays(nRays, centerPoint);

                for (auto &&p : *points)
                {
                    double bestDistance = std::numeric_limits<double>::max();
                    double bestProjectedLength = std::numeric_limits<double>::max();
                    size_t bestRayIndex = 0;

                    for (size_t r = 0; r < clusterRays.size(); r++)
                    {
                        const double projectedLength = clusterRays[r]->computeProjectedPointLength(data, p);
                        const double distance = clusterRays[r]->distanceToPoint(data, p, projectedLength);
                        if (distance < bestDistance)
                        {
                            bestDistance = distance;
                            bestProjectedLength = projectedLength;
                            bestRayIndex = r;
                        }
                    }

                    // Assign point to the ray with smallest distance.
                    clusterRays[bestRayIndex]->assign(p, bestDistance, bestProjectedLength);
                }
            }

            return rayContainer;
        }

        std::shared_ptr<blaze::DynamicVector<double>>
        calcCenter(const blaze::DynamicMatrix<double> &data, const std::vector<size_t> &pointIndicies) const
        {
            const size_t d = data.columns();

            auto center = std::make_shared<blaze::DynamicVector<double>>();
            center->resize(d);

            // Reset variables.
            center->reset();

            for (auto &&pointIndex : pointIndicies)
            {
                for (size_t j = 0; j < d; j++)
                {
                    center->at(j) += data.at(pointIndex, j);
                }
            }

            // Divide centers by the number of points in each cluster.
            double nElements = static_cast<double>(pointIndicies.size());
            for (size_t j = 0; j < d; j++)
            {
                center->at(j) /= nElements;
            }

            return center;
        }
    };
}
