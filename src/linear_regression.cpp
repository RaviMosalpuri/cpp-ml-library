#include "linear_regression.h"
#include <cmath>
#include <stdexcept>

LinearRegression::LinearRegression(const std::vector<double>& x, const std::vector<double>& y)
	:m_beta0(0.0), m_beta1(0.0), m_isSimple(true)
{
	if (x.size() != y.size())
	{
		throw std::invalid_argument("x and y must have the same size.");
	}

	size_t size = x.size();
	double x_sum = 0.0, y_sum = 0.0;
	
	for (double i : x) x_sum += i;
	for (double j : y) y_sum += j;

	double x_mean = x_sum / size, y_mean = y_sum / size;
	
	double num_sum = 0.0, den_sum = 0.0;
	for (size_t i = 0; i < size; ++i)
	{
		num_sum += (x[i] - x_mean) * (y[i] - y_mean);
		den_sum += pow((x[i] - x_mean), 2);
	}

	if (den_sum == 0.0)
	{
		throw std::runtime_error("Cannot compute regression: all x values are the same.");
	}

	// Compute beta1 and beta0 values
	// y = m_beta0 + m_beta1 * x
	m_beta1 = num_sum / den_sum;
	m_beta0 = y_mean - m_beta1 * x_mean;
}

LinearRegression::LinearRegression(const Matrix& X, const std::vector<double>& y)
	:m_beta(std::vector<double>(0)), m_beta0(0.0), m_beta1(0.0), m_isSimple(false)
{
	if (X.getNumOfRows() != y.size())
	{
		throw std::invalid_argument("Number of observations in X and y must match.");
	}
	else if (X.getNumOfCols() == 0)
	{
		throw std::invalid_argument("X cannot be empty.");
	}

	// Initialise the matrix with addition intercept (+1)
	Matrix X_with_intercept(X.getNumOfRows(), X.getNumOfCols() + 1);

	for (size_t i = 0; i < X.getNumOfRows(); ++i)
	{
		X_with_intercept(i, 0) = 1.0;
		for (size_t j = 0; j < X.getNumOfCols(); ++j)
		{
			X_with_intercept(i, j + 1) = X(i, j);
		}
	}

	// Normal equations: beta = (X^T X)^(-1) X^T y
	// X^T * X
	Matrix XTX = X_with_intercept.transpose() * X_with_intercept;
	// Inverse of X^T * X
	Matrix XTX_inv = XTX.inverse();
	// X^T * y
	std::vector<double> XTy = X_with_intercept.transpose() * y;
	// Get coefficients or beta
	m_beta = XTX_inv * XTy;
}

double LinearRegression::predict(double x) const
{
	// Return the predicted value
	return m_beta0 + m_beta1 * x;
}

double LinearRegression::predict(std::vector<double>& X) const
{
	if (m_isSimple)
	{
		throw std::invalid_argument("Use scalar predict for simple linear regression");
	}
	else if (X.size() != m_beta.size() - 1)
	{
		throw std::invalid_argument("Feature vector size must match number of coefficients (excluding intercept).");
	}

	// Prediction value equal to intercept
	double prediction = m_beta[0];

	for (size_t i = 0; i < X.size(); ++i)
	{
		// Get prediction value from input and coefficients
		prediction += m_beta[i+1] * X[i];
	}

	return prediction;
}

double LinearRegression::getIntercept() const
{
	// Return the intercept
	return m_beta0;
}

double LinearRegression::getSlope() const
{
	// Return the slope
	return m_beta1;
}

std::vector<double> LinearRegression::getCoefficients() const
{
	// Return the coefficients
	return m_beta;
}
