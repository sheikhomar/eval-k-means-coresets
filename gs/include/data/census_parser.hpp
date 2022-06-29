#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include <blaze/Math.h>
#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

#include <data/data_parser.hpp>

namespace data
{
    class CensusParser : public data::IDataParser
    {
    public:
        std::shared_ptr<blaze::DynamicMatrix<double>>
        parse(const std::string &filePath);
    };
}
