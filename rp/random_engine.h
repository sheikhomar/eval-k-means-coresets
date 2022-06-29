#ifndef RANDOMNESS_H
#define RANDOMNESS_H

#include <random>

class RandomEngine
{
public:

    static std::mt19937& get()
    {
        static std::mt19937 engine;
        return engine;
    }
};

#endif
