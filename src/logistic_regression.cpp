#include "logistic_regression.h"
#include "vector_utils.h"
#include <cmath>

LogisticRegression::LogisticRegression(double learningRate, size_t iterations)
	:m_learningRate(learningRate), m_iterations(iterations), m_weights(std::vector<double>(0)), m_data(nullptr)
{
}

void LogisticRegression::fit(const Matrix& X, const std::vector<double>& y)
{
	if (X.getNumOfRows() != y.size())
	{
		throw std::invalid_argument("Number of observations in X and y must match.");
	}

	size_t numOfRows = X.getNumOfRows(), numOfCols = X.getNumOfCols();
	m_weights = std::vector<double>(numOfCols + 1, 0.0);

	m_data = std::make_unique<Matrix>(numOfRows, numOfCols + 1);
	
	for (size_t i = 0; i < numOfRows; ++i)
	{
		(*m_data)(i,0) = 1.0;

		for (size_t j = 0; j < numOfCols; ++j)
		{
			(*m_data)(i, j + 1) = X(i, j);
		}
	}

	for (size_t k = 0; k < m_iterations; ++k)
	{
		const std::vector<double> predictions = sigmoid(*(m_data) * m_weights);
		const std::vector<double> errors = (predictions - y);
		const std::vector<double> gradients = ((*m_data).transpose() * errors) / numOfRows;
		m_weights = m_weights - (m_learningRate * gradients);
	}
}

int LogisticRegression::predict(const std::vector<double>& x) const
{
	if (x.size() != m_weights.size() - 1)
	{
		throw std::invalid_argument("Vector size is incorrect.");
	}

	// Initialise with bias value
	double result = m_weights[0];

	for (size_t i = 0; i < x.size(); ++i)
	{
		result += (x[i] * m_weights[i+1]);
	}
	
	double prob = sigmoid(result);
	return prob > 0.5 ? 1 : 0;
}

double LogisticRegression::sigmoid(double z) const
{
	// Return the sigmoid value of z
	return 1.0/(1.0 + std::exp(-z));
}

std::vector<double> LogisticRegression::sigmoid(std::vector<double> z) const
{
	std::vector<double> result(z.size(), 0.0);

	for (size_t i = 0; i < z.size(); ++i)
	{
		result[i] = sigmoid(z[i]);
	}

	return result;
}
