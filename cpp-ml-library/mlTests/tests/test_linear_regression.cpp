#include "linear_regression.h"
#include <gtest/gtest.h>
#include <vector>

// Test simple linear regression
TEST(LinearRegressionTest, SimpleLinearRegression)
{
    std::vector<double> x = { 1, 2, 3, 4 };
    std::vector<double> y = { 2, 3, 4, 5 };
    LinearRegression lr(x, y);
    EXPECT_DOUBLE_EQ(lr.getIntercept(), 1.0);
    EXPECT_DOUBLE_EQ(lr.getSlope(), 1.0);
    EXPECT_DOUBLE_EQ(lr.predict(5.0), 6.0);
}

// Test multiple linear regression
TEST(LinearRegressionTest, MultipleLinearRegression)
{
    // y = 1 + 2*x1 + 3*x2
    Matrix X(5, 2);
    X(0, 0) = 1; X(0, 1) = 1;  // x1, x2
    X(1, 0) = 2; X(1, 1) = 1;
    X(2, 0) = 3; X(2, 1) = 2;
    X(3, 0) = 4; X(3, 1) = 2;
    X(4, 0) = 5; X(4, 1) = 3;
    std::vector<double> y = { 6, 8, 13, 15, 20 };
    LinearRegression lr(X, y);
    std::vector<double> coeffs = lr.getCoefficients();
    EXPECT_NEAR(coeffs[0], 1.0, 1e-10);  // Intercept
    EXPECT_NEAR(coeffs[1], 2.0, 1e-10);  // Coefficient for x1
    EXPECT_NEAR(coeffs[2], 3.0, 1e-10);  // Coefficient for x2
    std::vector<double> test_input = { 5, 3 };
    EXPECT_NEAR(lr.predict(test_input), 20.0, 1e-10);  // 1 + 2*5 + 3*3
}

// Test invalid input handling
TEST(LinearRegressionTest, InvalidInput)
{
    std::vector<double> x = { 1, 2, 3 };
    std::vector<double> y = { 4, 5 };  // Size mismatch
    EXPECT_THROW(LinearRegression lr(x, y), std::invalid_argument);
}