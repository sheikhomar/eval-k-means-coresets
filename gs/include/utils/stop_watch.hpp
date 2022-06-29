#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <chrono>
#include <iomanip>

using namespace std::chrono;

namespace utils
{
    class StopWatch
    {
    private:
        system_clock::time_point startTime;
    
    public:
        StopWatch(bool startWatch = false);

        void start();

        std::string elapsedStr();
    };
}
