// #include <cxxopts.hpp>
#include <clustering/local_search.hpp>
#include <clustering/kmeans.hpp>
#include <clustering/kmeans1d.hpp>
#include <coresets/basic.hpp>
#include <coresets/sensitivity_sampling.hpp>
#include <coresets/group_sampling.hpp>
#include <coresets/stream_km.hpp>
#include <data/data_parser.hpp>
#include <data/bow_parser.hpp>
#include <data/csv_parser.hpp>
#include <data/census_parser.hpp>
#include <data/covertype_parser.hpp>
#include <data/tower_parser.hpp>
#include <utils/random.hpp>
#include <utils/stop_watch.hpp>
#include <coresets/ray.hpp>
#include <blaze/Blaze.h>

using namespace std;
using namespace clustering;
using namespace data;

void svd() 
{
  auto parser = CovertypeParser();
  utils::StopWatch timeDataParsing(true);
  auto data = parser.parse("data/input/covtype.data.gz");

  std::cout << "Data parsed: " << data->rows() << " x " << data->columns() << " in "<< timeDataParsing.elapsedStr() << ".\n";

  utils::StopWatch svdTime(true);
  //blaze::DynamicMatrix<double, blaze::rowMajor> A = *data;

  blaze::DynamicMatrix<double, blaze::rowMajor>     U;  // The matrix for the left singular vectors
  blaze::DynamicVector<double, blaze::columnVector> s;  // The vector for the singular values
  blaze::DynamicMatrix<double, blaze::rowMajor>     V;  // The matrix for the right singular vectors

  blaze::svd(*data, U, s, V);  // (3) Computing the singular values and vectors of A

  // Take the k singular vectors corresponding to the largest singular values
  // Zero out the rest of singular vectors on V
  // X_transformed = X*V*V^T

  std::cout << "SVD completed in "<< svdTime.elapsedStr() << ".\n";
  std::cout << " - Number of singular values: " << s.size() << "\n";
  std::cout << " - Shape of V: " << V.rows() << " x " << V.columns() << "\n";
  std::cout << "Singular values: \n" << s << "\n";

  std::cout << "V_1 :\n" << blaze::row(V, 0) << "\n";

  std::cout << "Storing SVD components..\n";
  blaze::Archive<std::ofstream> archive("/home/omar/code/coresets-bench/data/input/docword.enron.svd.blaze");
  archive << U << s << V;

  std::cout << "Done!";
}

void writeDoneFile(const std::string &outputDir)
{
    std::string outputFilePath = outputDir + "/done.out";
    std::ofstream outData(outputFilePath, std::ifstream::out);
    outData << "done\n";
    outData.close();
}

void outputResultsToFile(const std::shared_ptr<blaze::DynamicMatrix<double>> originalDataPoints, const std::shared_ptr<coresets::Coreset> coreset, const std::string &outputDir)
{
  std::string outputFilePath = outputDir + "/results.txt.gz";

  namespace io = boost::iostreams;
  std::ofstream fileStream(outputFilePath, std::ios_base::out | std::ios_base::binary);
  io::filtering_streambuf<io::output> fos;
  fos.push(io::gzip_compressor(io::gzip_params(io::gzip::best_compression)));
  fos.push(fileStream);
  std::ostream outData(&fos);

  coreset->writeToStream(*originalDataPoints, outData);
}

int main(int argc, char **argv)
{
  if (argc < 8)
  {
    std::cout << "Usage: algorithm dataset k m seed output_path [low_data_path]" << std::endl;
    std::cout << "  algorithm     = algorithm" << std::endl;
    std::cout << "  dataset       = dataset name" << std::endl;
    std::cout << "  data_path     = file path to dataset" << std::endl;
    std::cout << "  k             = number of desired centers" << std::endl;
    std::cout << "  m             = coreset size" << std::endl;
    std::cout << "  seed          = random seed" << std::endl;
    std::cout << "  output_dir    = path to output results" << std::endl;
    std::cout << std::endl;
    std::cout << "7 arguments expected, got " << argc - 1 << ":" << std::endl;
    for (int i = 1; i < argc; ++i)
      std::cout << " " << i << ": " << argv[i] << std::endl;
    return 1;
  }

  std::string algorithmName(argv[1]);
  std::string datasetName(argv[2]);
  std::string dataFilePath(argv[3]);
  size_t k = std::stoul(argv[4]);
  size_t m = std::stoul(argv[5]);
  int randomSeed = std::stoi(argv[6]);
  std::string outputDir(argv[7]);

  boost::algorithm::to_lower(algorithmName);
  boost::algorithm::trim(algorithmName);

  boost::algorithm::to_lower(datasetName);
  boost::algorithm::trim(datasetName);

  std::cout << "Running " << algorithmName << " with following parameters:\n";
  std::cout << " - Dataset:       " << datasetName << "\n";
  std::cout << " - Input path:    " << dataFilePath << "\n";
  std::cout << " - Clusters:      " << k << "\n";
  std::cout << " - Coreset size:  " << m << "\n";
  std::cout << " - Random Seed:   " << randomSeed << "\n";
  std::cout << " - Output dir:    " << outputDir << "\n";

  std::cout << "Initializing randomess with random seed: " << randomSeed << "\n";
  utils::Random::initialize(randomSeed);
  
  std::shared_ptr<IDataParser> dataParser;
  if (datasetName == "census")
  {
    dataParser = std::make_shared<CensusParser>();
  }
  else if (datasetName == "covertype")
  {
    dataParser = std::make_shared<CovertypeParser>();
  }
  else if (
    datasetName == "enron" ||
    datasetName == "nytimes"
  )
  {
    dataParser = std::make_shared<BagOfWordsParser>();
  }
  else if (datasetName == "tower")
  {
    dataParser = std::make_shared<TowerParser>();
  }
  else if (
    datasetName.find("hardinstance") != std::string::npos ||
    datasetName.find("lowd") != std::string::npos ||
    datasetName == "caltech101" ||
    datasetName == "nytimes100d"
  )
  {
    dataParser = std::make_shared<CsvParser>();
  }
  else
  {
    std::cout << "Unknown dataset: " << datasetName << "\n";
    return -1;
  }

  std::shared_ptr<blaze::DynamicMatrix<double>> data;
  {
    utils::StopWatch timeDataParsing(true);
    std::cout << "Parsing data:" << std::endl;
    data = dataParser->parse(dataFilePath);
    std::cout << "Data parsed: " << data->rows() << " x " << data->columns() << " in "<< timeDataParsing.elapsedStr() << std::endl;
  }

  std::cout << "Begin coreset algorithm: " << algorithmName << "\n";
  std::shared_ptr<coresets::Coreset> coreset;
  utils::StopWatch timeCoresetComputation(true);

  if (algorithmName == "basic-clustering")
  {
    coresets::BasicClustering algo(m);
    coreset = algo.run(*data);
  }
  else if (algorithmName == "stream-km++")
  {
    coresets::StreamKMeans algo(m);
    coreset = algo.run(*data);
  }
  else if (algorithmName == "sensitivity-sampling")
  {
    coresets::SensitivitySampling algo(2*k, m);
    coreset = algo.run(*data);
  }
  else if (algorithmName == "group-sampling")
  {
    size_t beta = 10000;
    size_t groupRangeSize = 4;
    size_t minimumGroupSamplingSize = 1;
    coresets::GroupSampling algo(2*k, m, beta, groupRangeSize, minimumGroupSamplingSize);
    coreset = algo.run(*data);
  }
  else if (algorithmName == "ray-maker")
  {
    size_t maxNumberOfRaysPerCluster = 20;
    coresets::RayMaker algo(2*k, m, maxNumberOfRaysPerCluster);
    coreset = algo.run(*data);
  }
  else 
  {
    std::cout << "Unknown algorithm: " << algorithmName << "\n";
    return -1;
  }
  
  std::cout << "Algorithm completed in " << timeCoresetComputation.elapsedStr() << std::endl;

  // if (useLowDimDataset)
  // {
  //   // We used low-dimensional data to compute the coreset.
  //   // Get rid of the low-dimensional data and load the original data.
  //   data->resize(0, 0, false);
  //   data->shrinkToFit();

  //   std::cout << "Parsing original data:\n";
  //   utils::StopWatch timeDataParsing(true);
  //   data = dataParser->parse(dataFilePath);
  //   std::cout << "Data parsed: " << data->rows() << " x " << data->columns() << " in "<< timeDataParsing.elapsedStr() << ".\n";
  // }

  outputResultsToFile(data, coreset, outputDir);
  writeDoneFile(outputDir);
  return 0;
}

int main_old() {
    blaze::DynamicMatrix<double> data {
    { -0.794152276623841F, 2.104951171962879F, },
    { -9.151551856068068F, -4.812864488195191F, },
    { -11.44182630908269F, -4.4578144103096555F, },
    { -9.767617768288718F, -3.19133736705118F, },
    { -4.536556476851341F, -8.401862882339504F, },
    { -6.263021151786394F, -8.1066608061999F, },
    { -6.384812343779634F, -8.473029703522716F, },
    { -9.204905637733754F, -4.5768792770429965F, },
    { -2.760179083161441F, 5.551213578682775F, },
    { -1.1710417594110225F, 4.330918155822106F, },
    { -10.036408012919152F, -5.5691209020665F, },
    { -9.875891232661665F, -2.823864639451285F, },
    { -7.175329210075055F, -8.770590168336406F, },
    { -2.406718199699357F, 6.098944469870908F, },
    { -4.874182454688006F, -10.049589027515138F, },
    { -6.078546995836497F, -7.939694203288603F, },
    { -6.832387624479001F, -7.47067669775956F, },
    { -2.346732606068119F, 3.561284227344442F, },
    { -10.341566179224177F, -3.9097516905289575F, },
    { -11.092624349394143F, -3.7839661143045364F, },
    { -6.502121087038712F, -7.912491012386313F, },
    { -10.263931009656723F, -3.920734000669846F, },
    { -6.816083022269968F, -8.449869256994909F, },
    { -1.340520809891421F, 4.157119493365752F, },
    { -10.372997453743215F, -4.592078954817427F, },
    { -7.374998957175799F, -10.588065868731183F, },
    { -6.623517740089157F, -8.253383337907545F, },
    { -1.359389585992692F, 4.054240022349643F, },
    { -0.19745196890354544F, 2.3463491593455075F, },
    { -6.5443058465843675F, -9.297569494247188F, },
    { -1.9274479855745354F, 4.9368453355813475F, },
    { -2.8020781039706595F, 4.057147146430284F, },
    { -7.581976641577967F, -9.150254932274308F, },
    { -1.8513954583101344F, 3.5188609047583252F, },
    { -8.370061750504195F, -3.615336850788729F, },
    { -7.251451955565088F, -8.25497397715319F, },
    { -8.798794623751593F, -3.7681921298792607F, },
    { -11.370829823899857F, -3.6381891553209127F, },
    { -10.178632805731251F, -4.557269175156462F, },
    { -7.2013269275537715F, -8.272282292398854F, },
    { -6.7842171065351F, -8.226340808371322F, },
    { -9.647166524988995F, -5.265631958600636F, },
    { -1.9819771099620271F, 4.022435514174746F, },
    { -11.227770639320063F, -3.402811051386989F, },
    { -9.799412783526332F, -3.834339901555746F, },
    { -6.5354168593050295F, -8.015526894626658F, },
    { -0.757969185355724F, 4.908984207745029F, },
    { 0.5260155005846419F, 3.009993533355024F, },
    { -2.7768702545837973F, 4.640905566660254F, },
    { -1.7824501314671677F, 3.4707204345840927F, },
    { -10.220040646263461F, -4.154106616293202F, },
    { -6.4058323875575285F, -9.780666445240302F, },
    { -6.987061055745032F, -7.5348478426255205F, },
    { -7.465760375446665F, -7.329222486173637F, },
    { -1.5394009534668904F, 5.023692978550581F, },
    { -6.569670859679778F, -8.327931264366546F, },
    { -10.617713347601232F, -3.255316513290986F, },
    { -8.723956573494325F, -1.98624679810847F, },
    { -1.6173461592329268F, 4.9893050825589835F, },
    { -1.1466300855305107F, 4.108397033740446F, },
    { -9.811151112664817F, -3.543296900154948F, },
    { -7.711798871912647F, -7.251741212975334F, },
    { -6.561697370222412F, -6.860002222091783F, },
    { -10.02232945952888F, -4.728510166532364F, },
    { -11.855694368099854F, -2.7171845169103843F, },
    { -5.733425071070147F, -8.440535968100065F, },
    { -2.4139578469451726F, 5.659358024076449F, },
    { -8.337440938903733F, -7.839680384160613F, },
    { -1.8319881134989553F, 3.5286314509217895F, },
    { -9.574218149588988F, -3.8760084790146454F, },
    { -9.5942208618623F, -3.3597700241261377F, },
    { -9.257156052556827F, -4.907049149171139F, },
    { -6.46256290336211F, -7.732945900976985F, },
    { -0.8205764920740146F, 5.337591950146718F, },
    { 0.00024227116135100424F, 5.148534029420497F, },
    { -9.682077556411496F, -5.975549763187208F, },
    { -6.195996026651871F, -7.402816464759037F, },
    { -7.02121319047935F, -8.379542347137651F, },
    { -2.187731658211975F, 3.333521246686991F, },
    { -10.4448410684391F, -2.7288408425577058F, },
    { -0.5279305184970926F, 5.92630668526536F, },
    { -11.196980535988288F, -3.090003229819183F, },
    { -9.837675434205272F, -3.0771796262469797F, },
    { -5.160223475316758F, -7.04217140606354F, },
    { -2.351220657673829F, 4.0097363419871845F, },
    { -0.5257904636130821F, 3.3065986015291307F, },
    { -1.4686444212810534F, 6.506745005322004F, },
    { -0.7587039566841077F, 3.7227620096688283F, },
    { -10.303916516281474F, -3.1253739047559583F, },
    { -2.3308060367853387F, 4.39382526992426F, },
    { -5.904543613663969F, -7.783735388248322F, },
    { -1.6087521511724905F, 3.769494222273808F, },
    { -1.8684541393232976F, 4.993113060025359F, },
    { -10.668374789942131F, -3.5757847610422853F, },
    { -8.876294795417436F, -3.544448009426377F, },
    { -6.026057581798325F, -5.966248457649787F, },
    { -7.047472775357734F, -9.275246833370932F, },
    { -1.3739725806942609F, 5.291631033113889F, },
    { -6.2539305108541825F, -7.108786009916786F, },
    { 0.08525185826796045F, 3.6452829679480585F, },
    { 1.0F, -10.0F, }, 
    { 6.0F, -16.0F, }, // Outlier point.
  };

  // kmeans::KMeans kMeansAlg(3, true, 100U, 0.0001);
  // auto result = kMeansAlg.run(data);
  // std::cout << "Cluster labels: \n" << result->getCentroids() <<  "\n" ;

  // coresets::SensitivitySampling sensitivitySampling;
  // sensitivitySampling.run(data);

  //LocalSearch ls(3, 2);
  //auto result = ls.run(data);

  // auto result = ls.runPlusPlus(data, 10, 100);
  // auto cost = (result->getClusterAssignments()).getTotalCost();
  // printf("Final cost: %0.5f\n", cost);

  // std::cout << "Final centers: \n" << result->getCentroids();

  // size_t k = 3; // Number of clusters.
  // size_t T = 20; // Number of target points.
  // size_t T_s = 1; //
  // size_t beta = 100; //Variable for ring ranges
  // size_t H = 4; // Group range size
  // coresets::GroupSampling gs(k, T, beta, H, T_s);
  // auto coreset = gs.run(data);

  // coresets::SensitivitySampling sensitivitySampling(k, T);
  // auto coreset = sensitivitySampling.run(data);

  // coresets::StreamKMeans streamKMeans(T);
  // auto coreset = streamKMeans.run(data);

  // for (size_t i = 0; i < coreset->size(); i++)
  // {
  //   auto point = coreset->at(i);
  //   printf(point->IsCenter ? "Center" : "Point");
  //   printf(" %ld with weight %0.4f\n", point->Index, point->Weight);
  // }

  // auto parser = CensusParser();
  // auto parsedData = parser.parse("data/input/USCensus1990.data.txt");
  

  // auto parser = CovertypeParser();
  // auto parsedData = parser.parse("data/input/covtype.data.gz");
  // size_t k = 10; // Number of clusters.
  // size_t T = 200*k; // Number of target points.
  // size_t maxNumberOfRaysPerCluster = 20;
  // coresets::RayMaker rayMaker(k, T, maxNumberOfRaysPerCluster);
  // auto coreset = rayMaker.run(*parsedData);

  // coresets::StreamKMeans algo(100);
  // auto coreset = algo.run(*parsedData);

  // outputResultsToFile(parsedData, coreset, "data");

  size_t k = 3; // Number of clusters.
  size_t T = 20; // Number of target points.
  size_t maxNumberOfRaysPerCluster = 20;
  coresets::RayMaker rayMaker(k, T, maxNumberOfRaysPerCluster);
  auto coreset = rayMaker.run(data);
  
  for (size_t i = 0; i < coreset->size(); i++)
  {
    auto point = coreset->at(i);
    printf(point->IsCenter ? "Center" : "Point");
    printf("! %ld with weight %0.4f\n", point->Index, point->Weight);
  }

  // std::cout << "Hello world!\n";

  // 
  // std::vector<double> y = {1.0};
  // size_t length = x.size();
  // // size_t k = 10;
  // size_t minK = k;
  // size_t maxK = k;
  // // const double * xp(x.data()), * yp (y.data());
  // // std::vector<int> clusters(x.size());
  // // std::string method = "linear";
  // // std::string estimateK = "BIC";
  // // std::vector<double> BIC(maxK - minK + 1);

  // // // cdef double [:] sizes = np.zeros(max_k, dtype=np.float64)
  // // std::vector<double> sizes(maxK);
  // // std::vector<double> centers(maxK);
  // // std::vector<double> withinss(maxK);
  
  // // double * center_p (centers.data()), * sp (sizes.data());
  // // double * bp (BIC.data()), * wp (withinss.data());
  // // int * cluster_p (clusters.data());

  // // kmeans_1d_dp(xp, length, yp, minK, maxK,
  // //             cluster_p, center_p, wp, sp, bp,
  // //             estimateK, method, L2);

  //std::vector<double> x = {-4, -5, -1, 0, 4, -4, -6, 7, 8, 10, 22};
  // std::vector<double> x = {4.0, 4.1, 4.2, -50.0, 200.2, 200.4, 200.9, 80, 100, 102};
  // k = 4;
  // size_t n = x.size();
  // std::vector<size_t> clusterLabels(n);
  // std::vector<double> centers(k);
  // clustering::kmeans1d::cluster(x, k, clusterLabels.data(), centers.data());

  // for (size_t i = 0; i < n; i++)
  // {
  //   std::cout << "Cluster for point " << i << ": " << clusterLabels[i] << std::endl;
  // }

  // for (size_t i = 0; i < k; i++)
  // {
  //   std::cout << "Center " << i << ": " << centers[i] << std::endl;
  // }
  std::cout << "Done!\n";

  return 0;
}
