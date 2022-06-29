#include <data/bow_parser.hpp>

using namespace data;
namespace io = boost::iostreams;

std::shared_ptr<blaze::DynamicMatrix<double>>
BagOfWordsParser::parse(const std::string &filePath)
{
    std::ifstream fileStream(filePath, std::ios_base::in | std::ios_base::binary);
    io::filtering_streambuf<io::input> filteredInputStream;
    if (boost::ends_with(filePath, ".gz"))
    {
        filteredInputStream.push(io::gzip_decompressor());
    }
    filteredInputStream.push(fileStream);
    std::istream inData(&filteredInputStream);

    // The format of the BoW files is 3 header lines, followed by data triples:
    // ---
    // D    -> the number of documents
    // W    -> the number of words in the vocabulary
    // NNZ  -> the number of nonzero counts in the bag-of-words
    // docID wordID count
    // docID wordID count
    // ...
    // docID wordID count
    // docID wordID count
    // ---

    std::string line;
    std::getline(inData, line); // Read line with D
    auto dataSize = std::stoul(line.c_str());
    std::getline(inData, line); // Read line with W
    auto dimSize = std::stoul(line.c_str());
    std::getline(inData, line); // Skip line with NNZ

    bool firstDataLine = true;
    size_t previousDocId = 0, currentRow = 0, docId = 0, wordId = 0;
    size_t lineNo = 3;
    double count;

    auto data = std::make_shared<blaze::DynamicMatrix<double>>(dataSize, dimSize);
    data->reset();

    while (inData.good())
    {
        std::getline(inData, line);
        lineNo++;

        std::vector<std::string> splits;
        boost::split(splits, line, boost::is_any_of(" "));

        if (splits.size() != 3)
        {
            // printf("Skipping line no %ld: '%s'.\n", lineNo, line.c_str());
            continue;
        }

        docId = std::stoul(splits[0]);
        wordId = std::stoul(splits[1]) - 1; // Convert to zero-based array indexing
        count = static_cast<double>(std::stoul(splits[2]));

        if (firstDataLine)
        {
            firstDataLine = false;
            previousDocId = docId;
        }

        if (previousDocId != docId)
        {
            currentRow++;
        }

        data->at(currentRow, wordId) = count;

        previousDocId = docId;
    }

    return data;
}