#include <data/covertype_parser.hpp>

using namespace data;
namespace io = boost::iostreams;

std::shared_ptr<blaze::DynamicMatrix<double>>
CovertypeParser::parse(const std::string &filePath)
{
    if (!boost::filesystem::exists(filePath))
    {
        std::stringstream errMsg;
        errMsg << "Unable to parse Covertype data because file " << filePath << " does not exist.";
        throw std::invalid_argument(errMsg.str());
    }

    std::ifstream fileStream(filePath, std::ios_base::in | std::ios_base::binary);
    io::filtering_streambuf<io::input> filteredInputStream;
    if (boost::ends_with(filePath, ".gz"))
    {
        filteredInputStream.push(io::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    auto dimSize = 54UL;
    auto dataSize = 581012UL;

    auto data = std::make_shared<blaze::DynamicMatrix<double>>(dataSize, dimSize);
    data->reset();
    
    size_t lineNo = 0, currentRow = 0;

    while (inData.good())
    {
        std::string line;
        std::getline(inData, line);
        lineNo++;

        std::vector<std::string> splits;
        boost::split(splits, line, boost::is_any_of(","));

        if (splits.size() != dimSize + 1)
        {
            continue;
        }

        // By looping up to dimSize, we skip the last attribute (class attribute).
        // This follows the StreamKM++ paper which removed the classification
        // attribute so in total they had 54 attributes.
        for (size_t j = 0; j < dimSize; j++)
        {
            data->at(currentRow, j) = atof(splits[j].c_str());
        }

        currentRow++;
    }

    if (currentRow < dataSize)
    {
        std::stringstream errMsg;
        errMsg << "Unexpected number of valid lines in " << filePath << ". Expected " << dataSize << " but read " << currentRow;
        throw std::length_error(errMsg.str());
    }

    return data;
}
