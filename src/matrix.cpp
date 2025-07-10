#include "matrix.h"

std::vector<double> Matrix::operator*(const std::vector<double>& vec) const
{
    if (vec.size() != m_numCols)
    {
        throw std::invalid_argument("Dimensions for Matrix-vector multiplication do not match.");
    }

    std::vector<double> result(m_numRows, 0.0);

    for (size_t i = 0; i < m_numRows; ++i)
    {
        for (size_t j = 0; j < m_numCols; ++j)
        {
            result[i] += m_data[i][j] * vec[j];
        }
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (m_numCols != other.m_numRows)
    {
        throw std::invalid_argument("Dimensions for Matrix-matrix multiplication do not match.");
    }

    Matrix result(m_numRows, other.m_numCols);

    for (size_t i = 0; i < m_numRows; ++i)
    {
        for (size_t j = 0; j < other.m_numCols; ++j)
        {
            for (size_t k = 0; k < m_numCols; ++k)
            {
                result.m_data[i][j] += m_data[i][k] * other.m_data[k][j];
            }
        }
    }

    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(m_numCols, m_numRows);

    for (size_t i = 0; i < m_numRows; ++i)
    {
        for (size_t j = 0; j < m_numCols; ++j)
        {
            result.m_data[j][i] = m_data[i][j];
        }
    }

    return result;
}

Matrix Matrix::inverse() const
{
    if (m_data.size() == 0)
    {
        throw std::invalid_argument("Matrix is empty.");
    }
    else if (m_numRows != m_numCols)
    {
        throw std::invalid_argument("Inverse is only defined for square matrix.");
    }

    // Initialise augmented matrix, matrix with data on left and identity matrix on right
    Matrix aug(m_numRows, 2 * m_numRows);
    
    for (size_t i = 0; i < m_numRows; ++i)
    {
        // Loop over the first part to get the data
        for (size_t j = 0; j < m_numCols; ++j)
        {
            aug.m_data[i][j] = m_data[i][j];
        }

        // Loop over the second part to get the identity matrix
        for (size_t j = m_numRows; j < 2 * m_numRows; ++j)
        {
            // Check for diagonal element
            if ((j - m_numRows) == i)
            {
                aug.m_data[i][j] = 1.0;
            }
        }
    }

    // Get the inverse of matrix using Gaussian Jordan elimination
    for (size_t i = 0; i < m_numRows; ++i)
    {
        double pivot = aug.m_data[i][i];

        if (pivot == 0.0)
        {
            throw std::invalid_argument("Inverse is not defined for singular matrix.");
        }

        for (size_t j = 0; j < 2 * m_numRows; ++j)
        {
            aug.m_data[i][j] /= pivot;
        }

        for (size_t k = 0; k < m_numRows; ++k)
        {
            // Check if row is same as pivot row
            if (k == i)
            {
                continue;
            }
            double factor = aug.m_data[k][i];

            for (size_t j = 0; j < 2 * m_numRows; ++j)
            {
                aug.m_data[k][j] -= factor * aug.m_data[i][j];
            }
        }
    }

    Matrix inverse(m_numRows, m_numRows);

    for (size_t i = 0; i < m_numRows; ++i)
    {
        for (size_t j = 0; j < m_numRows; ++j)
        {
            inverse.m_data[i][j] = aug.m_data[i][j + m_numRows];
        }
    }

    return inverse;
}
