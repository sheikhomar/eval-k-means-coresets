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
     * A coreset construction based on k-means++ initialisation.
     */
    class BasicClustering
    {
    public:
        /**
         * Number of points that the algorithm should aim to include in the coreset: T
         */
        const size_t TargetSamplesInCoreset;

        /**
         * Instantiate algorithm.
         */
        BasicClustering(size_t targetSamplesInCoreset);

        /**
         * Run algorithm.
         */
        std::shared_ptr<Coreset>
        run(const blaze::DynamicMatrix<double> &data);

    private:
        utils::Random random;
    };
}
