#include <utils/random.hpp>

using namespace utils;

static std::mt19937 randomEngine;

RandomIndexer::RandomIndexer(size_t s) : sampler(0, s - 1)
{
}

size_t
RandomIndexer::next()
{
    return sampler(randomEngine);
}

void
Random::initialize(int fixedSeed)
{
    if (fixedSeed == -1)
    {
        // Source: https://stackoverflow.com/questions/15509270/does-stdmt19937-require-warmup
        std::array<int, 624> seedData;
        std::random_device randomDevice;
        std::generate_n(seedData.data(), seedData.size(), std::ref(randomDevice));
        std::seed_seq randomSeq(std::begin(seedData), std::end(seedData));
        randomEngine.seed(randomSeq);
    }
    else
    {
        randomEngine.seed(static_cast<uint>(fixedSeed));
    }
}

RandomIndexer
Random::getIndexer(size_t size)
{
    return RandomIndexer(size);
}

double
Random::getDouble()
{
    return pickRandomValue(randomEngine);
}

Random::Random()
{
}

std::shared_ptr<blaze::DynamicVector<size_t>>
Random::runWeightedReservoirSampling(const size_t k, blaze::DynamicVector<double> &weights)
{
    size_t n = weights.size();

    auto indexSampler = this->getIndexer(k);
    auto data = std::make_shared<blaze::DynamicVector<size_t>>(k);
    data->reset();

    // Algorithm by M. T. Chao
    double sum = 0;

    // Fill the reservoir array
    for (size_t i = 0; i < k; i++)
    {
        (*data)[i] = i;
        sum = sum + weights[i];
    }

    double kDouble = static_cast<double>(k);

    for (size_t i = k; i < n; i++)
    {
        sum = sum + weights[i];

        // Compute the probability for item i
        double p_i = (kDouble * weights[i]) / sum;

        // Random value between 0 and 1
        auto q = this->getDouble();

        if (q <= p_i)
        {
            auto sampleIndex = indexSampler.next();
            (*data)[sampleIndex] = i;
        }
    }

    return data;
}

std::shared_ptr<blaze::DynamicVector<size_t>>
Random::choice(const size_t k, blaze::DynamicVector<double> &weights)
{
    auto result = std::make_shared<blaze::DynamicVector<size_t>>(k);
    result->reset();

    std::discrete_distribution<size_t> weightedChoice(weights.begin(), weights.end());

    for (size_t i = 0; i < k; i++)
    {
        size_t pickedIndex = weightedChoice(randomEngine);
        (*result)[i] = pickedIndex;
    }

    return result;
}

size_t
Random::choice(blaze::DynamicVector<double> &weights)
{
    std::discrete_distribution<size_t> weightedChoice(weights.begin(), weights.end());
    size_t pickedIndex = weightedChoice(randomEngine);
    return pickedIndex;
}

size_t
Random::stochasticRounding(double value)
{
    auto valueHigh = floor(value);
    auto valueLow = ceil(value);
    auto proba = (value - valueLow) / (valueHigh - valueLow);
    auto randomVal = this->getDouble();
    if (randomVal < proba)
    {
        return static_cast<size_t>(round(valueHigh)); // Round up
    }
    return static_cast<size_t>(round(valueLow)); // Round down
}

void Random::normal(blaze::DynamicVector<double> &vector)
{
    std::normal_distribution<double> distribution(0.0, 1.0);
    auto entryGenerator = [&distribution]() {
        return distribution(randomEngine);
    };
    std::generate(vector.begin(), vector.end(), entryGenerator);
}
