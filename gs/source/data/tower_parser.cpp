#include <data/tower_parser.hpp>

using namespace data;
namespace io = boost::iostreams;

std::shared_ptr<blaze::DynamicMatrix<double>>
TowerParser::parse(const std::string &filePath)
{
    std::ifstream fileStream(filePath, std::ios_base::in | std::ios_base::binary);
    io::filtering_streambuf<io::input> filteredInputStream;
    if (boost::ends_with(filePath, ".gz"))
    {
        filteredInputStream.push(io::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    auto dimSize = 3UL;
    auto dataSize = 4915200UL;

    auto data = std::make_shared<blaze::DynamicMatrix<double>>(dataSize, dimSize);
    data->reset();

    size_t lineNo = 0, currentRow = 0;

    while (inData.good())
    {
        for (size_t j = 0; j < dimSize; j++)
        {
            std::string line;
            std::getline(inData, line);
            lineNo++;
            data->at(currentRow, j) = static_cast<double>(std::stol(line));
        }
        
        currentRow++;
    }

    return data;
}
