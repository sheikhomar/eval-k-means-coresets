#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <blaze/Math.h>
#include <boost/array.hpp>

namespace utils
{

    class L2NormCalculator
    {
    private:
        const blaze::DynamicMatrix<double> &X;
        const blaze::DynamicMatrix<double> &Y;
        std::shared_ptr<std::vector<double>> xSquaredL2Norms;
        std::shared_ptr<std::vector<double>> ySquaredL2Norms;
        const bool ComputeSquared;
        
    public:
        L2NormCalculator(const blaze::DynamicMatrix<double> &x, bool computeSquaredL2Norm) : X(x), Y(x), ComputeSquared(computeSquaredL2Norm)
        {
            xSquaredL2Norms = std::make_shared<std::vector<double>>();
            xSquaredL2Norms->resize(x.rows());
            computeSquaredL2Norms(x, *xSquaredL2Norms);

            ySquaredL2Norms = xSquaredL2Norms;
        }

        L2NormCalculator(const blaze::DynamicMatrix<double> &x, const blaze::DynamicMatrix<double> &y, bool computeSquaredL2Norm) : X(x), Y(y), ComputeSquared(computeSquaredL2Norm)
        {
            xSquaredL2Norms = std::make_shared<std::vector<double>>();
            xSquaredL2Norms->resize(x.rows());
            computeSquaredL2Norms(x, *xSquaredL2Norms);

            ySquaredL2Norms = std::make_shared<std::vector<double>>();
            ySquaredL2Norms->resize(y.rows());
            computeSquaredL2Norms(y, *ySquaredL2Norms);
        }

        void computeSquaredL2Norms(const blaze::DynamicMatrix<double> &dataPoints, std::vector<double> &squaredNorms)
        {
            double val = 0.0;
            for (size_t i = 0; i < dataPoints.rows(); i++)
            {
                double sumOfSquares = 0.0;
                for (size_t j = 0; j < dataPoints.columns(); j++)
                {
                    val = dataPoints.at(i, j);
                    if (val != 0.0)
                    {
                        sumOfSquares += val * val;
                    }
                }
                squaredNorms[i] = sumOfSquares;
            }
        }

        double calc(size_t xIndex, size_t yIndex)
        {
            if (xIndex == yIndex)
            {
                return 0.0;
            }

            double dotProd = 0.0, val1 = 0.0, val2 = 0.0;
            for (size_t i = 0; i < X.columns(); i++)
            {
                val1 = X.at(xIndex, i);
                val2 = Y.at(yIndex, i);
                if (val1 != 0.0 && val2 != 0.0) // Only compute for non-zero
                {
                    dotProd += val1 * val2;
                }
            }
            double squaredNorm = (*xSquaredL2Norms)[xIndex] + (*ySquaredL2Norms)[yIndex] - 2 * dotProd;

            return (ComputeSquared) ? squaredNorm : blaze::sqrt(squaredNorm);
        }
    };
}
