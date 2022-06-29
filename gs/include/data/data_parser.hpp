#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include <blaze/Math.h>

namespace data
{
    /**
     * Represents a data parser.
     */
    class IDataParser
    {
    public:
        virtual ~IDataParser() {}

        /**
         * Parses the given file and converts it into a data matrix.
         */
        virtual std::shared_ptr<blaze::DynamicMatrix<double>>
        parse(const std::string &filePath) = 0; // pure virtual method
    };
}
