#include "matrix.h"

std::vector<double> Matrix::operator*(const std::vector<double>& x)
{
    return std::vector<double>();
}

Matrix Matrix::operator*(const Matrix& X)
{
    return Matrix(0,0);
}

Matrix Matrix::transpose() const
{
    Matrix result(_numCols, _numRows);
    for (size_t i = 0; i < _numRows; ++i)
    {
        for (size_t j = 0; j < _numCols; ++j)
        {
            result._data[j][i] = _data[i][j];
        }
    }

    return result;
}
