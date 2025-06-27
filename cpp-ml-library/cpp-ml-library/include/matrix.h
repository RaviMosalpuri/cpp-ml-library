#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>

/// <summary>
/// Class for the implementation of a Matrix
/// </summary>
class Matrix
{
public:
	/// <summary>
	/// Constructor for Matrix
	/// </summary>
	/// <param name="numRows">Number of rows</param>
	/// <param name="numCols">Number of columns</param>
	Matrix(size_t numRows, size_t numCols) 
		:m_numRows(numRows), m_numCols(numCols), m_data(numRows, std::vector<double>(numCols, 0.0)) { }

	/// <summary>
	/// Constructor for Matrix 
	/// </summary>
	/// <param name="data"></param>
	Matrix(const std::vector<std::vector<double>>& data)
		:m_numRows(data.size()), m_numCols(data.empty() ? 0 : data[0].size()), m_data(data) { }

	/// <summary>
	/// Overloaded for Matrix-vector multiplication
	/// </summary>
	/// <param name="x">Input vector</param>
	/// <returns>Resultant vector</returns>
	std::vector<double> operator*(const std::vector<double>& x);

	/// <summary>
	/// Overloaded for Matrix-Matrix multiplication
	/// </summary>
	/// <param name="X">Input matrix</param>
	/// <returns>Resultant matrix</returns>
	Matrix operator*(const Matrix& X);

	/// <summary>
	/// Transpose of Matrix
	/// </summary>
	/// <returns>Resultant matrix</returns>
	Matrix transpose() const;
private:
	// Number of rows
	size_t m_numRows;

	// Number of columns
	size_t m_numCols;

	// Matrix data
	std::vector<std::vector<double>> m_data;
};

#endif // !MATRIX_H