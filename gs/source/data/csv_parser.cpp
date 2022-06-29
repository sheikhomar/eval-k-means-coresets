#include <data/csv_parser.hpp>

using namespace data;
namespace io = boost::iostreams;
namespace x3 = boost::spirit::x3;

std::shared_ptr<blaze::DynamicMatrix<double>>
CsvParser::parse(const std::string &filePath)
{
    if (!boost::filesystem::exists(filePath))
    {
        std::stringstream errMsg;
        errMsg << "Unable to parse CSV data because file " << filePath << " does not exist.";
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

    auto data = std::make_shared<blaze::DynamicMatrix<double>>(0, 0);

    size_t lineNo = 0, currentRow = 0, dimSize = 0;
    size_t allocateNoOfRows = 100000;

    while (inData.good())
    {
        std::string line;
        std::getline(inData, line);
        lineNo++;

        if (line.size() < 1)
        {
            // "Skipping line no %ld: expected %ld values but empty line found.\n", lineNo, dimSize);
            continue;
        }

        std::vector<double> values;
        if (x3::phrase_parse(line.begin(), line.end(), (x3::double_ % ','), x3::space, values))
        {
            if (lineNo == 1)
            {
                dimSize = values.size();
                data->resize(allocateNoOfRows, dimSize);
            }

            if (values.size() != dimSize)
            {
                // "Skipping line no %ld: expected %ld values but got %ld.\n", lineNo, dimSize, values.size());
                continue;
            }

            if (currentRow >= data->rows())
            {
                data->resize(data->rows() + allocateNoOfRows, dimSize);
            }

            for (size_t j = 0; j < dimSize; j++)
            {
                data->at(currentRow, j) = values[j];
            }

            currentRow++;
        }
        else
        {
            std::stringstream errMsg;
            errMsg << "Failed to parse line " << lineNo;
            throw std::logic_error(errMsg.str());
        }
    }

    data->resize(currentRow, dimSize);
    data->shrinkToFit();

    return data;
}
