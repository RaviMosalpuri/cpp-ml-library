#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

class LinearRegression
{
public:
	// Constructor
	LinearRegression(const std::vector<double>& x, const std::vector<double>& y);

	/// <summary>
	/// Get the predicted value.
	/// </summary>
	/// <param name="x">Input value</param>
	/// <returns>Predicted value</returns>
	double predict(double x) const;

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

private:
	double beta0;
	double beta1;
};

#endif // !LINEAR_REGRESSION_H