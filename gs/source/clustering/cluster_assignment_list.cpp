#include <clustering/cluster_assignment_list.hpp>

using namespace clustering;

ClusterAssignmentList::ClusterAssignmentList(size_t n, size_t k) : numOfPoints(n), numOfClusters(k), clusters(n), distances(n)
{
}

ClusterAssignmentList::ClusterAssignmentList(const ClusterAssignmentList &other) : numOfPoints(other.numOfPoints), numOfClusters(other.numOfClusters), clusters(other.clusters), distances(other.distances)
{
}

void ClusterAssignmentList::assign(size_t pointIndex, size_t clusterIndex, double distance)
{
    // TODO: Ensure arguments are not out of range to avoid runtime errors.
    clusters[pointIndex] = clusterIndex;
    distances[pointIndex] = distance;
}

void ClusterAssignmentList::assignAll(const blaze::DynamicMatrix<double> &dataPoints, const blaze::DynamicMatrix<double> &centers)
{
    auto n = this->numOfPoints;
    auto k = this->numOfClusters;

    // For each data point, assign the centroid that is closest to it.
    for (size_t p = 0; p < n; p++)
    {
        double bestDistance = std::numeric_limits<double>::max();
        size_t bestCluster = 0;

        // Loop through all the clusters.
        for (size_t c = 0; c < k; c++)
        {
            // Compute the L2 norm between point p and centroid c.
            const double distance = blaze::norm(blaze::row(dataPoints, p) - blaze::row(centers, c));

            // Decide if current distance is better.
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestCluster = c;
            }
        }

        // Assign cluster to the point p.
        this->assign(p, bestCluster, bestDistance);
    }
}

size_t
ClusterAssignmentList::getCluster(size_t pointIndex) const
{
    // TODO: Ensure arguments are not out of range to avoid runtime errors.
    return clusters[pointIndex];
}

size_t
ClusterAssignmentList::getNumberOfPoints() const
{
    return this->numOfPoints;
}

size_t
ClusterAssignmentList::getNumberOfClusters() const
{
    return this->numOfClusters;
}

blaze::DynamicVector<double> &
ClusterAssignmentList::getCentroidDistances()
{
    return this->distances;
}

size_t
ClusterAssignmentList::countPointsInCluster(size_t clusterIndex) const
{
    size_t count = 0;
    for (size_t p = 0; p < this->numOfPoints; p++)
    {
        if (clusters[p] == clusterIndex)
        {
            count++;
        }
    }

    return count;
}

double
ClusterAssignmentList::getTotalCost() const
{
    return blaze::sum(this->distances);
}

double
ClusterAssignmentList::getPointCost(size_t pointIndex) const
{
    // TODO: Ensure pointIndex is not out of bounds.

    return this->distances[pointIndex];
}

std::shared_ptr<blaze::DynamicVector<double>>
ClusterAssignmentList::calcAverageClusterCosts() const
{
    auto results = std::make_shared<blaze::DynamicVector<double>>(this->numOfClusters);
    results->reset();

    blaze::DynamicVector<double> counts(this->numOfClusters);

    for (size_t p = 0; p < this->numOfPoints; p++)
    {
        auto c = clusters[p];
        (*results)[c] += distances[p];
        counts[c] += 1;
    }

    for (size_t c = 0; c < this->numOfClusters; c++)
    {
        (*results)[c] /= counts[c];
    }

    return results;
}

std::shared_ptr<blaze::DynamicVector<double>>
ClusterAssignmentList::calcClusterCosts() const
{
    auto results = std::make_shared<blaze::DynamicVector<double>>(this->numOfClusters);
    results->reset();

    for (size_t p = 0; p < this->numOfPoints; p++)
    {
        auto c = clusters[p];
        (*results)[c] += distances[p];
    }

    return results;
}

ClusterAssignmentList &
ClusterAssignmentList::operator=(const ClusterAssignmentList &other)
{
    this->numOfPoints = other.numOfPoints;
    this->numOfClusters = other.numOfClusters;
    this->clusters = other.clusters;
    this->distances = other.distances;
    return *this;
}

blaze::DynamicVector<double>
ClusterAssignmentList::getNormalizedCosts() const
{
    return distances / this->getTotalCost();
}

size_t
getIndex(const std::vector<size_t> &values, const size_t valueToSearch)
{
    auto it = std::find(values.begin(), values.end(), valueToSearch);
    if (it != values.end())
    {
        auto index = it - values.begin();
        if (index > 0)
        {
            return static_cast<size_t>(index);
        }
    }

    throw std::logic_error("Value not found.");
}

void ClusterAssignmentList::calcCenters(const blaze::DynamicMatrix<double> &dataPoints, blaze::DynamicMatrix<double> &newCenters)
{
    assert(dataPoints.rows() == this->numOfPoints);

    const size_t n = dataPoints.rows();
    const size_t d = dataPoints.columns();
    const size_t k = this->numOfClusters;

    if (newCenters.rows() != k or newCenters.columns() != d)
    {
        newCenters.resize(n, d);
    }

    // Reset variables.
    newCenters = 0;
    std::vector<size_t> clusterMemberCounts(k);
    for (size_t c = 0; c < k; c++)
    {
        clusterMemberCounts[c] = 0;
    }

    auto centerIndices = *this->getClusterIndices();

    for (size_t p = 0; p < n; p++)
    {
        const size_t c = getIndex(centerIndices, this->clusters[p]);

        // Sum all points assigned in cluster c
        blaze::row(newCenters, c) += blaze::row(dataPoints, p);

        // Count number of points in each cluster.
        clusterMemberCounts[c] += 1;
    }

    // Divide centers by the number of points in each cluster.
    for (size_t c = 0; c < k; c++)
    {
        const auto count = std::max<size_t>(1, clusterMemberCounts[c]);
        blaze::row(newCenters, c) /= count;
    }
}

std::shared_ptr<blaze::DynamicVector<double>>
ClusterAssignmentList::calcCenter(const blaze::DynamicMatrix<double> &dataPoints, const size_t clusterIndex) const
{
    const size_t n = dataPoints.rows();
    const size_t d = dataPoints.columns();

    auto center = std::make_shared<blaze::DynamicVector<double>>();
    center->resize(d);

    // Reset variables
    center->reset();
    double memberCount = 0.0;

    for (size_t p = 0; p < n; p++)
    {
        const size_t c = this->clusters[p];

        if (c == clusterIndex)
        {
            for (size_t j = 0; j < d; j++)
            {
                center->at(j) += dataPoints.at(p, j);
            }

            memberCount += 1.0;
        }
    }

    // Divide centers by the number of points in each cluster.
    for (size_t j = 0; j < d; j++)
    {
        center->at(j) /= memberCount;
    }

    return center;
}

std::shared_ptr<std::vector<size_t>>
ClusterAssignmentList::getPointsByCluster(size_t clusterIndex) const
{
    auto pointsToReturn = std::make_shared<std::vector<size_t>>();
    for (size_t p = 0; p < this->numOfPoints; p++)
    {
        if (clusters[p] == clusterIndex)
        {
            pointsToReturn->push_back(p);
        }
    }
    return pointsToReturn;
}

std::shared_ptr<std::vector<size_t>>
ClusterAssignmentList::getClusterIndices() const
{
    auto clusterIndexSet = std::set<size_t>();
    for (size_t p = 0; p < this->numOfPoints; p++)
    {
        clusterIndexSet.insert(clusters[p]);
    }

    // Convert set to a vector.
    auto clusterIndices = std::make_shared<std::vector<size_t>>(clusterIndexSet.begin(), clusterIndexSet.end());
    return clusterIndices;
}
