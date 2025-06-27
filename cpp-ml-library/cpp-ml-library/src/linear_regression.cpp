#include "linear_regression.h"
#include <cmath>
#include <stdexcept>

LinearRegression::LinearRegression(const std::vector<double>& x, const std::vector<double>& y)
	:m_beta0(0.0), m_beta1(0.0), m_isSimple(true)
{
	// Compare size of inputs
	if (x.size() != y.size())
	{
		// Throw invalid argument exception
		throw std::invalid_argument("x and y must have the same size");
	}

	// Get the size of input
	size_t size = x.size();
	// Initialize sum to 0
	double x_sum = 0.0, y_sum = 0.0;
	
	// Get sum of all the x and y values
	for (double i : x) x_sum += i;
	for (double j : y) y_sum += j;

	// Get the mean of x and y values
	double x_mean = x_sum / size, y_mean = y_sum / size;
	
	// Initialize numerator and denominator to 0
	double num_sum = 0.0, den_sum = 0.0;
	for (size_t i = 0; i < size; ++i)
	{
		// Get the numerator and denominator values
		num_sum += (x[i] - x_mean) * (y[i] - y_mean);
		den_sum += pow((x[i] - x_mean), 2);
	}

	// Check if denominator is 0
	if (den_sum == 0.0)
	{
		// Throw runtime-error exception
		throw std::runtime_error("Cannot compute regression: all x values are the same");
	}

	// Compute beta1 and beta0 values
	m_beta1 = num_sum / den_sum;
	m_beta0 = y_mean - m_beta1 * x_mean;
}

LinearRegression::LinearRegression(const Matrix& X, const std::vector<double>& y)
	:m_beta(std::vector<double>(0)), m_beta0(0.0), m_beta1(0.0), m_isSimple(false)
{
}

double LinearRegression::predict(double x) const
{
	// Return the predicted value
	return m_beta0 + m_beta1 * x;
}

std::vector<double> LinearRegression::predict(std::vector<double>& X) const
{
	throw std::logic_error("Function not implemented yet");
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
