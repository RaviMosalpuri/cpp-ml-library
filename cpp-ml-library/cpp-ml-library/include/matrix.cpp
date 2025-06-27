#include "matrix.h"

std::vector<double> Matrix::operator*(const std::vector<double>& x)
{
    // Check if the vector size is not equal to number of columns
    if (x.size() != m_numCols)
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
            result[i] += m_data[i][j] * x[j];
        }
    }

    // Return the result
    return result;
}

Matrix Matrix::operator*(const Matrix& X)
{
    throw std::logic_error("Function not implemented yet");
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
