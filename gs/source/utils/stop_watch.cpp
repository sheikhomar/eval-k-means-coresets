#include <utils/stop_watch.hpp>

using namespace utils;

StopWatch::StopWatch(bool startWatch)
{
    if (startWatch)
    {
        start();
    }
}

void StopWatch::start()
{
    startTime = high_resolution_clock::now();
}

std::string StopWatch::elapsedStr()
{
    typedef std::chrono::duration<int, std::ratio<86400>> days;

    auto stop = high_resolution_clock::now();
    auto durationMs = duration_cast<milliseconds>(stop - startTime);

    auto durationDays = duration_cast<days>(durationMs);
    durationMs -= durationDays;

    auto durationHours = duration_cast<hours>(durationMs);
    durationMs -= durationHours;

    auto durationMins = duration_cast<minutes>(durationMs);
    durationMs -= durationMins;

    auto durationSecs = duration_cast<seconds>(durationMs);
    durationMs -= durationSecs;

    auto dayCount = durationDays.count();
    auto hourCount = durationHours.count();
    auto minCount = durationMins.count();
    auto secCount = durationSecs.count();
    auto msCount = durationMs.count();

    std::stringstream output;
    output.fill('0');

    if (dayCount)
    {
        output << dayCount << "d";
    }
    if (dayCount || hourCount)
    {
        if (dayCount)
        {
            output << " ";
        }
        output << std::setw(2) << hourCount << "h";
    }
    if (dayCount || hourCount || minCount)
    {
        if (dayCount || hourCount)
        {
            output << " ";
        }
        output << std::setw(2) << minCount << "m";
    }
    if (dayCount || hourCount || minCount || secCount)
    {
        if (dayCount || hourCount || minCount)
        {
            output << " ";
        }
        output << std::setw(2) << secCount << "s";
    }
    if (dayCount || hourCount || minCount || secCount || msCount)
    {
        if (dayCount || hourCount || minCount || secCount)
        {
            output << " ";
        }
        output << std::setw(3) << msCount << "ms";
    }

    return output.str();
}
