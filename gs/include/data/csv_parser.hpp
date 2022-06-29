#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>

#include <blaze/Math.h>
#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/spirit/home/x3.hpp>
#include <boost/spirit/include/support_istream_iterator.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <data/data_parser.hpp>

namespace data
{
    class CsvParser : public data::IDataParser
    {
    public:
        std::shared_ptr<blaze::DynamicMatrix<double>>
        parse(const std::string &filePath);
    };
}
