#include "logistic_regression.h"
#include "vector_utils.h"
#include <cmath>

LogisticRegression::LogisticRegression(double learningRate, size_t iterations)
	:m_learningRate(learningRate), m_iterations(iterations), m_weights(std::vector<double>(0)), m_data(nullptr)
{
}

void LogisticRegression::fit(const Matrix& X, const std::vector<double>& y)
{
	// Check if size of input features and labels match
	if (X.getNumOfRows() != y.size())
	{
		// Throw invalid argument exception
		throw std::invalid_argument("Number of observations in X and y must match.");
	}

	// Get number of rows and columns
	size_t numOfRows = X.getNumOfRows(), numOfCols = X.getNumOfCols();

	// Initialise weights, with one additional for bias
	m_weights = std::vector<double>(numOfCols + 1, 0.0);

	// Initialise the data
	m_data = std::make_unique<Matrix>(numOfRows, numOfCols + 1);
	
	// Loop over the rows
	for (size_t i = 0; i < numOfRows; ++i)
	{
		// Set bias term to 1
		(*m_data)(i,0) = 1.0;
		// Loop over the columns
		for (size_t j = 0; j < numOfCols; ++j)
		{
			// Get the values from features
			(*m_data)(i, j + 1) = X(i, j);
		}
	}

	// Loop till number of iterations
	for (size_t k = 0; k < m_iterations; ++k)
	{
		// Get the predictions
		const std::vector<double> predictions = sigmoid(*(m_data) * m_weights);
		// Get the error values
		const std::vector<double> errors = (predictions - y);
		// Get the gradient values
		const std::vector<double> gradients = ((*m_data).transpose() * errors) / numOfRows;
		// Update the weights
		m_weights = m_weights - (m_learningRate * gradients);
	}
}

int LogisticRegression::predict(const std::vector<double>& x) const
{
	// Check for the input feature size
	if (x.size() != m_weights.size() - 1)
	{
		// Throw invalid argument exception
		throw std::invalid_argument("Vector size is incorrect.");
	}

	// Initialise with bias value
	double result = m_weights[0];

	// Loop over features
	for (size_t i = 0; i < x.size(); ++i)
	{
		// Multiply element-wise with weights
		result += (x[i] * m_weights[i+1]);
	}
	
	// Get the probability from sigmoid
	double prob = sigmoid(result);
	// Return 1 if probability greater than 0.5, else return 0
	return prob > 0.5 ? 1 : 0;
}

double LogisticRegression::sigmoid(double z) const
{
	// Return the sigmoid value of z
	return 1.0/(1.0 + std::exp(-z));
}

std::vector<double> LogisticRegression::sigmoid(std::vector<double> z) const
{
	// Initialise result
	std::vector<double> result(z.size(), 0.0);

	// Loop over values
	for (size_t i = 0; i < z.size(); ++i)
	{
		// Store sigmoid values in result
		result[i] = sigmoid(z[i]);
	}

	// Return the result
	return result;
}
