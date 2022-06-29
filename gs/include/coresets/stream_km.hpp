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
    class StreamKMeans
    {
    public:
        /**
         * Number of points that the algorithm should aim to include in the coreset: T
         */
        const size_t TargetSamplesInCoreset;

        StreamKMeans(size_t targetSamplesInCoreset);

        std::shared_ptr<Coreset>
        run(const blaze::DynamicMatrix<double> &data);

    private:
        utils::Random random;
    };
}
