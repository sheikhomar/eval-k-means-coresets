#include <data/census_parser.hpp>

using namespace data;
namespace io = boost::iostreams;

std::shared_ptr<blaze::DynamicMatrix<double>>
CensusParser::parse(const std::string &filePath)
{
    std::ifstream fileStream(filePath, std::ios_base::in | std::ios_base::binary);
    io::filtering_streambuf<io::input> filteredInputStream;
    if (boost::ends_with(filePath, ".gz"))
    {
        filteredInputStream.push(io::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    std::string line;

    std::getline(inData, line); // Ignore the header line.
    // Preparing Census Dataset. Skip first line: %s\n", line.c_str());

    auto dimSize = 68UL;
    auto dataSize = 2458285UL;

    size_t currentRow = 0;
    size_t lineNo = 3;

    auto data = std::make_shared<blaze::DynamicMatrix<double>>(dataSize, dimSize);
    data->reset();

    while (inData.good())
    {
        std::getline(inData, line);
        lineNo++;

        std::vector<std::string> splits;
        boost::split(splits, line, boost::is_any_of(","));

        if (splits.size() != dimSize+1)
        {
            // printf("Skipping line no %ld: expected %ld values but got %ld.\n", lineNo, dimSize+1, splits.size());
            continue;
        }

        for (size_t j = 0; j < dimSize; j++)
        {
            // Skip the first attribute `caseid`
            data->at(currentRow, j) = atof(splits[j+1].c_str());
        }

        currentRow++;
    }

    return data;
}
