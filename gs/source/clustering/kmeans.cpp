#include <clustering/kmeans.hpp>

using namespace clustering;

KMeans::KMeans(size_t k, bool useKmeansPlusPlus, size_t nIter, double convDiff) : NumOfClusters(k), InitKMeansPlusPlus(useKmeansPlusPlus), MaxIterations(nIter), ConvergenceDiff(convDiff)
{
}

std::shared_ptr<ClusteringResult>
KMeans::run(const blaze::DynamicMatrix<double> &data)
{
  std::vector<size_t> initialCenters;

  if (this->InitKMeansPlusPlus)
  {
    auto clusters = this->pickInitialCentersViaKMeansPlusPlus(data);
    initialCenters = *clusters->getClusterIndices();
  }
  else
  {
    utils::Random random;

    auto randomPointGenerator = random.getIndexer(data.rows());

    for (size_t c = 0; c < this->NumOfClusters; c++)
    {
      // Pick a random point p as a cluster center.
      auto randomPoint = randomPointGenerator.next();
      initialCenters.push_back(randomPoint);
    }
  }

  auto centers = copyRows(data, initialCenters);
  return this->runLloydsAlgorithm(data, centers);
}

blaze::DynamicMatrix<double>
KMeans::copyRows(const blaze::DynamicMatrix<double> &data, const std::vector<size_t> &indicesToCopy)
{
  size_t k = indicesToCopy.size();
  size_t d = data.columns();

  blaze::DynamicMatrix<double> centers(k, d);
  for (size_t c = 0; c < k; c++)
  {
    size_t pointIndex = indicesToCopy[c];
    blaze::row(centers, c) = blaze::row(data, pointIndex);
  }
  return centers;
}

void computeSquaredNorms(const blaze::DynamicMatrix<double> &dataPoints, std::vector<double> &squaredNorms)
{
  double val = 0.0;
  for (size_t i = 0; i < dataPoints.rows(); i++)
  {
    double sumOfSquares = 0.0;
    for (size_t j = 0; j < dataPoints.columns(); j++)
    {
      val = dataPoints.at(i, j);
      if (val != 0.0)
      {
        sumOfSquares += val * val;
      }
    }
    squaredNorms[i] = sumOfSquares;
  }
}

std::shared_ptr<clustering::ClusterAssignmentList>
KMeans::pickInitialCentersViaKMeansPlusPlus(const blaze::DynamicMatrix<double> &data)
{
  utils::Random random;
  size_t n = data.rows();
  size_t k = this->NumOfClusters;
  utils::StopWatch sw(true);

  utils::L2NormCalculator squaredL2Norm(data, true);

  auto clusters = std::make_shared<clustering::ClusterAssignmentList>(n, k);

  // Declare an array used to maintain the squared distances of
  // every point to the closest center among the set of centers
  // that are being considered for any `c \in {2, 3, ..., k}`.
  // Whenever a new center is picked, we can compare the distance
  // between each point `p1` and the new center to determine whether
  // the array at index `p1` should be updated.
  blaze::DynamicVector<double> smallestDistances(n);
  for (size_t p1 = 0; p1 < n; p1++)
  {
    smallestDistances[p1] = std::numeric_limits<double>::max();
  }

  size_t centerIndex = 0;
  for (size_t c = 0; c < k; c++)
  {
    utils::StopWatch pickCenterSW(true);

    if (c == 0)
    {
      // Pick the first centroid uniformly at random.
      auto randomPointGenerator = random.getIndexer(n);
      centerIndex = randomPointGenerator.next();
    }
    else
    {
      for (size_t p1 = 0; p1 < n; p1++)
      {
        // Compute dist2(p, C_c) i.e., the squared distance between the point `p1` and 
        // the center `centerIndex` that we picked at the previous iteration.
        double distance = squaredL2Norm.calc(p1, centerIndex);

        // Compute min_dist^2(p, C_c-1)
        // Decide if the current distance is better.
        if (distance < smallestDistances[p1])
        {
          // Set the weight of a given point to be the smallest distance
          // to any of the previously selected center points. 
          smallestDistances[p1] = distance;
          clusters->assign(p1, centerIndex, distance);
        }
      }

      // Pick the index of a point randomly selected based on the distances.
      // A point with a large distance is more likely to be picked than one with
      // a small distance. We want to select points randomly such that points
      // that are far from any of the selected center points have higher likelihood of
      // being picked as the next candidate center.
      centerIndex = random.choice(smallestDistances);
    }
  }

  // Final reassignment step.
  for (size_t p1 = 0; p1 < n; p1++)
  {
    double distance = squaredL2Norm.calc(p1, centerIndex);
    if (distance < smallestDistances[p1])
    {
      clusters->assign(p1, centerIndex, distance);
    }
  }

  return clusters;
}

std::shared_ptr<ClusteringResult>
KMeans::runLloydsAlgorithm(const blaze::DynamicMatrix<double> &matrix, blaze::DynamicMatrix<double> centroids)
{
  const size_t n = matrix.rows();
  const size_t d = matrix.columns();
  const size_t k = this->NumOfClusters;

  blaze::DynamicVector<size_t> clusterMemberCounts(k);
  ClusterAssignmentList cal(n, k);

  if (MaxIterations == 0)
  {
    cal.assignAll(matrix, centroids);
  }

  if (MaxIterations > 0)
  {
    std::vector<double> dataSquaredNorms;
    dataSquaredNorms.resize(n);
    computeSquaredNorms(matrix, dataSquaredNorms);

    std::vector<double> centerSquaredNorms;
    centerSquaredNorms.resize(centroids.rows());
    computeSquaredNorms(centroids, centerSquaredNorms);

    // Lambda function computes the squared L2 distance between any pair of points.
    // The function will automatically use any precomputed distance if it exists.
    auto calcSquaredL2Norm = [&matrix, &centroids, d, &dataSquaredNorms, &centerSquaredNorms](size_t p, size_t c) -> double
    {
      double dotProd = 0.0;
      for (size_t i = 0; i < d; i++)
      {
        dotProd += matrix.at(p, i) * centroids.at(c, i);
      }

      return dataSquaredNorms[p] + centerSquaredNorms[c] - 2 * dotProd;
    };

    for (size_t i = 0; i < this->MaxIterations; i++)
    {
      utils::StopWatch iterSW(true);
      // For each data point, assign the centroid that is closest to it.
      for (size_t p = 0; p < n; p++)
      {
        double bestDistance = std::numeric_limits<double>::max();
        size_t bestCluster = 0;

        // Loop through all the clusters.
        for (size_t c = 0; c < k; c++)
        {
          // Compute the L2 norm between point p and centroid c.
          // const double distance = blaze::norm(blaze::row(matrix, p) - blaze::row(centroids, c));
          const double distance = calcSquaredL2Norm(p, c);

          // Decide if current distance is better.
          if (distance < bestDistance)
          {
            bestDistance = distance;
            bestCluster = c;
          }
        }

        // Assign cluster to the point p.
        cal.assign(p, bestCluster, blaze::sqrt(bestDistance));
      }

      // Move centroids based on the cluster assignments.

      // First, save a copy of the centroids matrix.
      blaze::DynamicMatrix<double> oldCentrioids(centroids);

      // Set all elements to zero.
      centroids = 0;           // Reset centroids.
      clusterMemberCounts = 0; // Reset cluster member counts.

      for (size_t p = 0; p < n; p++)
      {
        const size_t c = cal.getCluster(p);
        blaze::row(centroids, c) += blaze::row(matrix, p);
        clusterMemberCounts[c] += 1;
      }

      for (size_t c = 0; c < k; c++)
      {
        const auto count = std::max<size_t>(1, clusterMemberCounts[c]);
        blaze::row(centroids, c) /= count;
      }

      // Recompute the squared distances again.
      computeSquaredNorms(centroids, centerSquaredNorms);

      // Compute the Frobenius norm
      auto diffAbsMatrix = blaze::abs(centroids - oldCentrioids);
      auto diffAbsSquaredMatrix = blaze::pow(diffAbsMatrix, 2); // Square each element.
      auto frobeniusNormDiff = blaze::sqrt(blaze::sum(diffAbsSquaredMatrix));

      if (frobeniusNormDiff < this->ConvergenceDiff)
      {
        break;
      }
    }
  }

  return std::make_shared<ClusteringResult>(cal, centroids);
}
