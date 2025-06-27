#include "matrix.h"

std::vector<double> Matrix::operator*(const std::vector<double>& vec) const
{
    // Check if the vector size is not equal to number of columns
    if (vec.size() != m_numCols)
    {
        // Throw invalid argument
        throw std::invalid_argument("Dimensions for Matrix-vector multiplication do not match.");
    }

    // Initialise the resultant vector with size = number of rows
    std::vector<double> result(m_numRows, 0.0);

    // Loop over the rows
    for (size_t i = 0; i < m_numRows; ++i)
    {
        // Loop over the columns
        for (size_t j = 0; j < m_numCols; ++j)
        {
            // Store the element-wise multiplication in result
            result[i] += m_data[i][j] * vec[j];
        }
    }

    // Return the result
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    // Check if the matrix columns size is not equal to number of columns
    if (m_numCols != other.m_numRows)
    {
        // Throw invalid argument
        throw std::invalid_argument("Dimensions for Matrix-matrix multiplication do not match.");
    }

    // Initialise the resultant matrix with size = number of rows x number of columns
    Matrix result(m_numRows, other.m_numCols);

    // Loop over the rows
    for (size_t i = 0; i < m_numRows; ++i)
    {
        // Loop over the other matrix columns
        for (size_t j = 0; j < other.m_numCols; ++j)
        {
            // Loop over the columns
            for (size_t k = 0; k < m_numCols; ++k)
            {
                // Store the element-wise multiplication in result
                result.m_data[i][j] += m_data[i][k] * other.m_data[k][j];
            }
        }
    }

    // Return the result
    return result;
}

Matrix Matrix::transpose() const
{
    // Initialise the resultant Matrix with size = number of columns x number of rows
    Matrix result(m_numCols, m_numRows);

    // Loop over the rows
    for (size_t i = 0; i < m_numRows; ++i)
    {
        // Loop over the columns
        for (size_t j = 0; j < m_numCols; ++j)
        {
            // Get the resultant data
            result.m_data[j][i] = m_data[i][j];
        }
    }

    // Return the result
    return result;
}

Matrix Matrix::inverse() const
{
    // Check if the size of matrix is 0
    if (m_data.size() == 0)
    {
        // Throw matrix empty exception
        throw std::invalid_argument("Matrix is empty.");
    }
    // Check if number of rows is equal to columns, matrix should be square matrix
    else if (m_numRows != m_numCols)
    {
        // Throw inverse not defined exception
        throw std::invalid_argument("Inverse is only defined for square matrix.");
    }

    // Initialise augmented matrix, matrix with data on left and identity matrix on right
    Matrix aug(m_numRows, 2 * m_numRows);
    
    // Loop over the rows
    for (size_t i = 0; i < m_numRows; ++i)
    {
        // Loop over the first part to get the data
        for (size_t j = 0; j < m_numCols; ++j)
        {
            // Add data to augmented matrix
            aug.m_data[i][j] = m_data[i][j];
        }

        // Loop over the second part to get the identity matrix
        for (size_t j = m_numRows; j < 2 * m_numRows; ++j)
        {
            // Check for diagonal element
            if ((j - m_numRows) == i)
            {
                // Add data to augmented matrix
                aug.m_data[i][j] = 1.0;
            }
        }
    }

    // Get the inverse of matrix using Gaussian Jordan elimination
    // Loop over the rows
    for (size_t i = 0; i < m_numRows; ++i)
    {
        // Get the pivot element
        double pivot = aug.m_data[i][i];

        // Check if pivot is 0
        if (pivot == 0.0)
        {
            // Throw invalid argument exception
            throw std::invalid_argument("Inverse is not defined for singular matrix.");
        }

        // Loop over the columns
        for (size_t j = 0; j < 2 * m_numRows; ++j)
        {
            // Divide the numbers by pivot
            aug.m_data[i][j] /= pivot;
        }

        // Loop over the rows
        for (size_t k = 0; k < m_numRows; ++k)
        {
            // Check if row is same as pivot row
            if (k == i)
            {
                // Continue
                continue;
            }

            // Get the factor from augmented matrix
            double factor = aug.m_data[k][i];

            // Loop over the columns
            for (size_t j = 0; j < 2 * m_numRows; ++j)
            {
                // Multiply every element by factor then subtract
                aug.m_data[k][j] -= factor * aug.m_data[i][j];
            }
        }
    }

    // Initialise the inverse matrix with size as number of rows * number of rows
    Matrix inverse(m_numRows, m_numRows);

    // Loop over the rows
    for (size_t i = 0; i < m_numRows; ++i)
    {
        // Loop over the columns
        for (size_t j = 0; j < m_numRows; ++j)
        {
            // Get the inverse matrix data from the second part of augmented matrix
            inverse.m_data[i][j] = aug.m_data[i][j + m_numRows];
        }
    }

    // Return the inverse matrix
    return inverse;
}
