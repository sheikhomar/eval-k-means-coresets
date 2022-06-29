#ifndef MATRICES_H_INCLUDED
#define MATRICES_H_INCLUDED

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <cstdlib>
#include <stdexcept>
#include <map>
#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include <boost/core/noncopyable.hpp>
#include "robin_hood.h"
#include "stop_watch.h"

class DokSparseMatrix : boost::noncopyable
{
public:

private:
    size_t _nRows;
    size_t _nColumns;
    size_t _rowPower;
    robin_hood::unordered_node_map<uint64_t, double> data;

public:
    DokSparseMatrix() { }

    void setSize(size_t nRows, size_t nColumns)
    {
        _nRows = nRows;
        _nColumns = nColumns;
        _rowPower = static_cast<size_t>(ceilf64(log2f64(nRows)));
    }

    uint64_t inline getKey(const size_t rowIndex, const size_t columnIndex) const
    {
        // Maps two indices into one deterministically. We could use Cantor or Szudzik pairing functions:
        // https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/
        // Instead we opt for the simple solution as we already know the matrix size.
        return (rowIndex << _rowPower) + columnIndex;
    }

    void set(const size_t rowIndex, const size_t columnIndex, double value)
    {
        auto key = getKey(rowIndex, columnIndex);
        data.emplace(key, value);
    }

    double at(size_t rowIndex, size_t columnIndex) const
    {
        auto key = getKey(rowIndex, columnIndex);
        if (data.find(key) != data.end())
        {
            return data.at(key);
        }

        return 0.0;
    }
    
    double getValue(size_t rowIndex, size_t columnIndex)
    {
        auto key = getKey(rowIndex, columnIndex);
        return data[key];
    }

    size_t rows() const { return this->_nRows; }
    size_t columns() const { return this->_nColumns; }
    size_t nnz() const { return this->data.size(); }
        
};

class CooSparseMatrix : boost::noncopyable
{
private:
    size_t _nRows;
    size_t _nColumns;
    size_t _rowPower;
    std::vector<size_t> _rowIndices;
    std::vector<size_t> _columnIndices;
    std::vector<double> _values;

public:
    CooSparseMatrix() { }

    void setSize(size_t nRows, size_t nColumns)
    {
        _nRows = nRows;
        _nColumns = nColumns;
    }

    void set(const size_t rowIndex, const size_t columnIndex, double value)
    {
        _rowIndices.push_back(rowIndex);
        _columnIndices.push_back(columnIndex);
        _values.push_back(value);
    }

    void printRowValues(size_t rowIndex) const
    {
        std::cout << "Row " << rowIndex << std::endl;

        for (size_t i = 0; i < _rowIndices.size(); i++)
        {
            if (_rowIndices[i] == rowIndex)
            {
                std::cout << "  Column " << _columnIndices[i] << " = " << _values[i]  << std::endl;
            }
        }
    }

    size_t rows() const { return this->_nRows; }
    size_t columns() const { return this->_nColumns; }
    size_t nnz() const { return this->_values.size(); }
    const std::vector<size_t>& rowIndices() const { return _rowIndices; }
    const std::vector<size_t>& columnIndices() const { return _columnIndices; }
    const std::vector<double>& values() const { return _values; }
};



class CsrMatrix : boost::noncopyable
{
private:
    size_t _nRows;
    size_t _nColumns;
    size_t _rowPower;
    std::vector<size_t> rowIndexPointers;
    std::vector<size_t> columnIndices;
    std::vector<double> values;

public:
    CsrMatrix() { }

    CsrMatrix(const CooSparseMatrix &cooMatrix)
    {
        _nRows = cooMatrix.rows();
        _nColumns = cooMatrix.columns();
        const auto nnz = cooMatrix.nnz();
        rowIndexPointers.resize(_nRows + 1);
        values.resize(nnz);
        columnIndices.resize(nnz);

        // Reset point index.
        for (size_t i = 0; i < _nRows; i++)
        {
            rowIndexPointers[i] = 0;
        }
        
        // Count number of non-zero column values in each row.
        for (size_t nzi = 0; nzi < nnz; nzi++)
        {
            const auto cooRowIndex = cooMatrix.rowIndices()[nzi];
            rowIndexPointers[cooRowIndex]++;
        }
        
        // Compute commulative sum of non-zero values.
        size_t cumsum = 0;
        for (size_t i = 0; i < _nRows; i++)
        {
            size_t nnzColumnValues = rowIndexPointers[i];
            rowIndexPointers[i] = cumsum;
            cumsum += nnzColumnValues;
        }

        rowIndexPointers[_nRows] = nnz;

        for (size_t nzi = 0; nzi < nnz; nzi++)
        {
            const auto cooRowIndex = cooMatrix.rowIndices()[nzi];
            const auto dest = rowIndexPointers[cooRowIndex];

            columnIndices[dest] = cooMatrix.columnIndices()[nzi];
            values[dest] = cooMatrix.values()[nzi];

            // Assumes that column indices are sorted.
            rowIndexPointers[cooRowIndex]++;
        }

        // Restore index pointers.
        for (size_t i = 0, last = 0; i <= _nRows; i++)
        {
            size_t temp = rowIndexPointers[i];
            rowIndexPointers[i]  = last;
            last   = temp;
        }
    }

    double at(size_t rowIndex, size_t columnIndex) const
    {
        auto nzStart = rowIndexPointers[rowIndex];
        auto nzEnd = rowIndexPointers[rowIndex+1];

        for (size_t ni = nzStart; ni < nzEnd; ni++)
        {
            if (columnIndices[ni] == columnIndex)
            {
                return values[ni];
            }
        }

        return 0.0;
    }

    std::pair<
        std::vector<size_t>::const_iterator, 
        std::vector<size_t>::const_iterator
    >
    getColumnIndicesForRow(size_t rowIndex) const
    {
        auto nzStart = rowIndexPointers[rowIndex];
        auto nzEnd = rowIndexPointers[rowIndex+1];

        return std::make_pair(columnIndices.begin() + nzStart, columnIndices.begin() + nzEnd);
    }

    /**
     * Perform following computation in sparse matrices:
     * 
     * for (j = 0; j < cols; j++)
     *   sketch[otherRow, j] += sign * data[ownRow, j];
     */
    void rowProductWithSign(const size_t ownRowIndex, const double sign, std::map<size_t, double> &otherRow) const
    {
        auto nzStart = rowIndexPointers[ownRowIndex];
        auto nzEnd = rowIndexPointers[ownRowIndex+1];
        size_t columnIndex = 0;
        double oldValue = 0.0;

        for (size_t ni = nzStart; ni < nzEnd; ni++)
        {
            columnIndex = columnIndices[ni];
            
            auto searchResult = otherRow.find(columnIndex);
            if (searchResult == otherRow.end())
            {
                // If column does not exist then create one
                otherRow.emplace(columnIndex, sign * values[ni]);
            } 
            else 
            {
                // If column exists then add to it.
                oldValue = searchResult->second;
                otherRow.at(columnIndex) = oldValue + sign * values[ni];
            }
        }
    }

    void printRowValues(size_t rowIndex) const
    {
        auto nzStart = rowIndexPointers[rowIndex];
        auto nzEnd = rowIndexPointers[rowIndex+1];

        std::cout << "Row " << rowIndex << std::endl;

        for (size_t ni = nzStart; ni < nzEnd; ni++)
        {
            std::cout << "  Column " << columnIndices[ni] << " = " << values[ni]  << std::endl;
        }
    }

    size_t rows() const { return this->_nRows; }
    size_t columns() const { return this->_nColumns; }
    size_t nnz() const { return this->values.size(); }
};


class SparseMatrix : boost::noncopyable
{
public:

private:
    size_t _nRows;
    size_t _nColumns;
    size_t _rowPower;
    robin_hood::unordered_node_map<uint64_t, double> data;
    std::vector<size_t> rowIndices;
    std::vector<size_t> columnIndices;
    std::vector<double> values;

public:
    SparseMatrix() { }

    void setSize(size_t nRows, size_t nColumns, size_t nonZeros)
    {
        // data = std::make_shared<ublas::compressed_matrix<double>>(nRows, nColumns, nonZeros);

        _nRows = nRows;
        _nColumns = nColumns;
        _rowPower = static_cast<size_t>(ceilf64(log2f64(nRows)));

        rowIndices.reserve(nRows + 1);
        if (nonZeros > 0)
        {
            values.reserve(nonZeros);
            columnIndices.reserve(nonZeros);
        }
        
    }

    uint64_t getKey(const size_t rowIndex, const size_t columnIndex) const
    {
        // Maps two indices into one deterministically. We could use Cantor or Szudzik pairing functions:
        // https://www.vertexfragment.com/ramblings/cantor-szudzik-pairing-functions/
        // Instead we opt for the simple solution as we already know the matrix size.
        return (rowIndex << _rowPower) + columnIndex;
    }

    void set(const size_t rowIndex, const size_t columnIndex, double value)
    {
        auto key = getKey(rowIndex, columnIndex);
        data.emplace(key, value);
        //data[key] = value;
        // (*data)(rowIndex, columnIndex) = value;
        // if (value == 0.0)
        // {
        //     if (data.find(key) != data.end())
        //     {
        //         data.erase(key);
        //     }
        // }
        // else
        // {
        //      data[key] = value;
        // }
    }

    double at(size_t rowIndex, size_t columnIndex) const
    {
        // return (*data)(rowIndex, columnIndex);
        auto key = getKey(rowIndex, columnIndex);
        if (data.find(key) != data.end())
        {
            return data.at(key);
        }

        return 0.0;
        
    }
    
    double getValue(size_t rowIndex, size_t columnIndex)
    {
        auto key = getKey(rowIndex, columnIndex);
        return data[key];
        // return (*data)(rowIndex, columnIndex);
    }

    size_t rows() const { return this->_nRows; }
    size_t columns() const { return this->_nColumns; }
    // size_t nnz() const { return this->data->nnz(); }
    size_t nnz() const { return this->data.size(); }
        
};

class Matrix : boost::noncopyable
{
private:
    double *_entries;
    size_t _nRows;
    size_t _nColumns;
    size_t _totalSize;
    bool allocated = false;

public:
    Matrix() {}
    Matrix(size_t rows, size_t columns)
    {
        allocate(rows, columns);
    }

    void deallocate()
    {
        if (this->allocated)
        {
            std::free(this->_entries);
            this->_nRows = 0;
            this->_nColumns = 0;
            this->_totalSize = 0;
            this->allocated = false;
        }
    }

    void allocate(size_t rows, size_t columns)
    {
        deallocate();

        std::cout << "Allocing memory for matrix: " << rows << "x" << columns << ".\n";
        size_t totalSize = rows * columns * sizeof(double);
        this->_entries = reinterpret_cast<double *>(std::calloc(rows * columns, sizeof(double)));
        this->_nRows = rows;
        this->_nColumns = columns;
        this->_totalSize = totalSize;
        this->allocated = true;
    }

    void set(size_t rowIndex, size_t columnIndex, double value)
    {
        size_t index = rowIndex * this->_nColumns + columnIndex;
        if (index > this->_totalSize)
        {
            throw std::invalid_argument("Index out of bounds.");
        }
        this->_entries[index] = value;
    }

    double at(size_t rowIndex, size_t columnIndex) const
    {
        size_t index = rowIndex * this->_nColumns + columnIndex;
        if (index > this->_totalSize)
        {
            throw std::invalid_argument("Index out of bounds.");
        }
        return this->_entries[index];
    }

    double *data() const { return this->_entries; }
    size_t rows() const { return this->_nRows; }
    size_t columns() const { return this->_nColumns; }
    size_t size() const { return this->_totalSize; }

    ~Matrix()
    {
        deallocate();
    }
};


void testCooToCsr()
{
    CooSparseMatrix data;
    data.setSize(7, 5);
    data.set(0, 0, 8);
    data.set(0, 2, 2);
    data.set(1, 2, 5);
    data.set(4, 2, 7);
    data.set(4, 3, 1);
    data.set(4, 4, 2);
    data.set(6, 3, 9);

    CsrMatrix csrMatrix(data);

    size_t columnIndex = 0;
    for (size_t i = 0; i < 7; i++)
    {
        std::cout << "Column index for row " << i << ": ";
        auto iterPair = csrMatrix.getColumnIndicesForRow(i);
        for (auto it = iterPair.first; it != iterPair.second; ++it)
        {
            columnIndex = *it;
            std::cout << columnIndex << "  ";
        }
        std::cout << "\n";
    }
}

#endif