#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "matrix.h"
#include <memory>
#include <vector>

/// <summary>
/// Class for implementation of Logistic Regression
/// </summary>
class LogisticRegression
{
public:
	/// <summary>
	/// Constructor for logistic regression.
	/// </summary>
	/// <param name="learningRate">Learning rate</param>
	/// <param name="iterations">Number of iterations</param>
	LogisticRegression(double learningRate = 0.01, size_t iterations = 1000);

	/// <summary>
	/// Fit the logistic regression
	/// </summary>
	/// <param name="X">Input features</param>
	/// <param name="y">Input labels</param>
	void fit(const Matrix& X, const std::vector<double>& y);

	/// <summary>
	/// Predict value from input
	/// </summary>
	/// <param name="x">Input features</param>
	/// <returns>Predicted value</returns>
	int predict(const std::vector<double>& x) const;

	/// <summary>
	/// Sigmoid function
	/// </summary>
	/// <param name="z">Input value</param>
	/// <returns>Resultant value</returns>
	double sigmoid(double z) const;

	/// <summary>
	/// Sigmoid function
	/// </summary>
	/// <param name="z">Input vector</param>
	/// <returns>Resultant vector</returns>
	std::vector<double> sigmoid(std::vector<double> z) const;
private:
	// Weights or coefficients
	std::vector<double> m_weights;

	// Learning rate
	double m_learningRate;

	// Number of iterations
	size_t m_iterations;

	// Data
	std::unique_ptr<Matrix> m_data;
};

#endif // !LOGISTIC_REGRESSION_H
