#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>

class Matrix
{
public:
	Matrix(size_t numRows, size_t numCols) 
		:_numRows(numRows), _numCols(numCols), _data(numRows, std::vector<double>(numCols, 0.0)) { }

	Matrix(const std::vector<std::vector<double>>& data)
		:_numRows(data.size()), _numCols(data.empty() ? 0 : data[0].size()), _data(data) { }

	std::vector<double> operator*(const std::vector<double>& x);

	Matrix operator*(const Matrix& X);

	Matrix transpose() const;
private:
	size_t _numRows;
	size_t _numCols;
	std::vector<std::vector<double>> _data;
};

#endif // !MATRIX_H