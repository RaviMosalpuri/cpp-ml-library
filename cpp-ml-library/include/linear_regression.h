#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>
#include "matrix.h"

/// <summary>
/// Class for implementation of Linear Regression
/// </summary>
class LinearRegression
{
public:
	/// <summary>
	/// Constructor for simple linear regression.
	/// </summary>
	/// <param name="x"></param>
	/// <param name="y"></param>
	LinearRegression(const std::vector<double>& x, const std::vector<double>& y);

	/// <summary>
	/// Constructor for multiple linear regression.
	/// </summary>
	/// <param name="X"></param>
	/// <param name="y"></param>
	LinearRegression(const Matrix& X, const std::vector<double>& y);

	/// <summary>
	/// Get the predicted value for simple linear regression.
	/// </summary>
	/// <param name="x">Input value</param>
	/// <returns>Predicted value</returns>
	double predict(double x) const;

	/// <summary>
	/// Get the predicted value for multiple linear regression.
	/// </summary>
	/// <param name="X">Input values</param>
	/// <returns>Predicted value</returns>
	double predict(std::vector<double>& X) const;

	/// <summary>
	/// Get the intercept.
	/// </summary>
	/// <returns>Intercept</returns>
	double getIntercept() const;

	/// <summary>
	/// Get the slope.
	/// </summary>
	/// <returns>Slope</returns>
	double getSlope() const;

	/// <summary>
	/// Returns the coefficients for multiple linear regression.
	/// </summary>
	/// <returns>Coefficients</returns>
	std::vector<double> getCoefficients() const;

private:
	// Intercept value for simple linear regression
	double m_beta0;

	// Slope value for simple linear regression
	double m_beta1;

	// Coefficient values for multiple linear regression
	std::vector<double> m_beta;

	// Boolean value for, is simple linear regression
	bool m_isSimple;
};

#endif // !LINEAR_REGRESSION_H