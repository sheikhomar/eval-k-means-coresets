#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace std;

// Original source: https://github.com/dstein64/kmeans1d/blob/master/kmeans1d/_core.cpp

namespace clustering::kmeans1d
{
    /*
    *  Internal implementation of the SMAWK algorithm.
    */
    template <typename T>
    void _smawk(
            const vector<size_t>& rows,
            const vector<size_t>& cols,
            const function<T(size_t, size_t)>& lookup,
            vector<size_t>* result) {
        // Recursion base case
        if (rows.size() == 0) return;

        // ********************************
        // * REDUCE
        // ********************************

        vector<size_t> _cols;  // Stack of surviving columns
        for (size_t col : cols) {
            while (true) {
                if (_cols.size() == 0) break;
                size_t row = rows[_cols.size() - 1];
                if (lookup(row, col) >= lookup(row, _cols.back()))
                    break;
                _cols.pop_back();
            }
            if (_cols.size() < rows.size())
                _cols.push_back(col);
        }

        // Call recursively on odd-indexed rows
        vector<size_t> odd_rows;
        for (size_t i = 1; i < rows.size(); i += 2) {
            odd_rows.push_back(rows[i]);
        }
        _smawk(odd_rows, _cols, lookup, result);

        unordered_map<size_t, size_t> col_idx_lookup;
        for (size_t idx = 0; idx < _cols.size(); ++idx) {
            col_idx_lookup[_cols[idx]] = idx;
        }

        // ********************************
        // * INTERPOLATE
        // ********************************

        // Fill-in even-indexed rows
        size_t start = 0;
        for (size_t r = 0; r < rows.size(); r += 2) {
            size_t row = rows[r];
            size_t stop = _cols.size() - 1;
            if (r < rows.size() - 1)
                stop = col_idx_lookup[(*result)[rows[r + 1]]];
            size_t argmin = _cols[start];
            T min = lookup(row, argmin);
            for (size_t c = start + 1; c <= stop; ++c) {
                T value = lookup(row, _cols[c]);
                if (c == start || value < min) {
                    argmin = _cols[c];
                    min = value;
                }
            }
            (*result)[row] = argmin;
            start = stop;
        }
    }

    /*
    *  Interface for the SMAWK algorithm, for finding the minimum value in each row
    *  of an implicitly-defined totally monotone matrix.
    */
    template <typename T>
    vector<size_t> smawk(
            const size_t num_rows,
            const size_t num_cols,
            const function<T(size_t, size_t)>& lookup) {
        vector<size_t> result;
        result.resize(num_rows);
        vector<size_t> rows(num_rows);
        iota(begin(rows), end(rows), 0);
        vector<size_t> cols(num_cols);
        iota(begin(cols), end(cols), 0);
        _smawk<T>(rows, cols, lookup, &result);
        return result;
    }

    /*
    *  Calculates cluster costs in O(1) using prefix sum arrays.
    */
    class CostCalculator {
        vector<double> cumsum;
        vector<double> cumsum2;

    public:
        CostCalculator(const vector<double>& vec, size_t n) {
            cumsum.push_back(0.0);
            cumsum2.push_back(0.0);
            for (size_t i = 0; i < n; ++i) {
                double x = vec[i];
                cumsum.push_back(x + cumsum[i]);
                cumsum2.push_back(x * x + cumsum2[i]);
            }
        }

        double calc(size_t i, size_t j) {
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wconversion"

            if (j < i) return 0.0;
            double mu = (cumsum[j + 1] - cumsum[i]) / (j - i + 1UL);
            double result = cumsum2[j + 1] - cumsum2[i];
            result += (j - i + 1UL) * (mu * mu);
            result -= (2 * mu) * (cumsum[j + 1] - cumsum[i]);

            #pragma GCC diagnostic pop

            return result;
        }
    };

    template <typename T>
    class Matrix {
        vector<T> data;
        size_t num_rows;
        size_t num_cols;

    public:
        Matrix(size_t nRows, size_t nCols) {
            this->num_rows = nRows;
            this->num_cols = nCols;
            data.resize(nRows * nCols);
        }

        inline T get(size_t i, size_t j) {
            return data[i * num_cols + j];
        }

        inline void set(size_t i, size_t j, T value) {
            data[i * num_cols + j] = value;
        }
    };

    void cluster(
            const std::vector<double> &array,
            const size_t k,
            size_t* clusters,
            double* centroids) {
        // ***************************************************
        // * Sort input array and save info for de-sorting
        // ***************************************************

        size_t n = array.size();

        vector<size_t> sort_idxs(n);
        iota(sort_idxs.begin(), sort_idxs.end(), 0);
        sort(
            sort_idxs.begin(),
            sort_idxs.end(),
            [&array](size_t a, size_t b) {return array[a] < array[b];});
        vector<size_t> undo_sort_lookup(n);
        vector<double> sorted_array(n);
        for (size_t i = 0; i < n; ++i) {
            sorted_array[i] = array[sort_idxs[i]];
            undo_sort_lookup[sort_idxs[i]] = i;
        }

        // ***************************************************
        // * Set D and T using dynamic programming algorithm
        // ***************************************************

        // Algorithm as presented in section 2.2 of (Grønlund et al., 2017).

        CostCalculator cost_calculator(sorted_array, n);
        Matrix<double> D(k, n);
        Matrix<size_t> T(k, n);

        for (size_t i = 0; i < n; ++i) {
            D.set(0, i, cost_calculator.calc(0, i));
            T.set(0, i, 0);
        }

        for (size_t k_ = 1; k_ < k; ++k_) {
            auto C = [&D, &k_, &cost_calculator](size_t i, size_t j) -> double {
                size_t col = i < j - 1 ? i : j - 1;
                return D.get(k_ - 1, col) + cost_calculator.calc(j, i);
            };
            vector<size_t> row_argmins = smawk<double>(n, n, C);
            for (size_t i = 0; i < row_argmins.size(); ++i) {
                size_t argmin = row_argmins[i];
                double min = C(i, argmin);
                D.set(k_, i, min);
                T.set(k_, i, argmin);
            }
        }

        // ***************************************************
        // * Extract cluster assignments by backtracking
        // ***************************************************

        // TODO: This step requires O(kn) memory usage due to saving the entire
        //       T matrix. However, it can be modified so that the memory usage is O(n).
        //       D and T would not need to be retained in full (D already doesn't need
        //       to be fully retained, although it currently is).
        //       Details are in section 3 of (Grønlund et al., 2017).

        vector<double> sorted_clusters(n);

        size_t t = n;
        size_t k_ = k - 1;
        size_t n_ = n - 1;
        // The do/while loop was used in place of:
        //   for (k_ = k - 1; k_ >= 0; --k_)
        // to avoid wraparound of an unsigned type.
        do {
            size_t t_ = t;
            t = T.get(k_, n_);
            double centroid = 0.0;
            for (size_t i = t; i < t_; ++i) {
                sorted_clusters[i] = static_cast<double>(k_);
                centroid += (sorted_array[i] - centroid) / static_cast<double>(i - t + 1);
            }
            centroids[k_] = centroid;
            k_ -= 1;
            n_ = t - 1;
        } while (t > 0);

        // ***************************************************
        // * Order cluster assignments to match de-sorted
        // * ordering
        // ***************************************************

        for (size_t i = 0; i < n; ++i) {
            clusters[i] = sorted_clusters[undo_sort_lookup[i]];
        }
    }

}
