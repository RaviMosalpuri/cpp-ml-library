#include <gtest/gtest.h>
#include "logistic_regression.h"
#include "matrix.h"
#include <vector>
#include <stdexcept>
#include <cmath>

// Test fixture for LogisticRegression
class LogisticRegressionTest : public ::testing::Test {
protected:
    LogisticRegression lr;
    LogisticRegressionTest() : lr(0.01, 1000) {}  // Learning rate 0.01, 1000 iterations
};

// Test fit and predict with 1D data
TEST_F(LogisticRegressionTest, FitAndPredict1D) {
    // 1D linearly separable data: X = [-2, -1, 0, 1, 2], y = [0, 0, 0, 1, 1]
    std::vector<std::vector<double>> X_data = { {-2}, {-1}, {0}, {1}, {2} };
    Matrix X(X_data);
    std::vector<double> y = { 0, 0, 0, 1, 1 };

    lr.fit(X, y);

    // Verify predictions
    EXPECT_EQ(lr.predict({ -1 }), 0);
    EXPECT_EQ(lr.predict({ 1 }), 1);
}

// Test fit and predict with 2D data
TEST_F(LogisticRegressionTest, FitAndPredict2D) {
    // 2D data: class 1 if y > x, else class 0
    std::vector<std::vector<double>> X_data = { {0,1}, {1,0}, {2,3}, {3,2}, {1,1}, {1,2}, {2,1} };
    Matrix X(X_data);
    std::vector<double> y = { 1, 0, 1, 0, 0, 1, 0 };

    lr.fit(X, y);

    // Verify predictions
    EXPECT_EQ(lr.predict({ 0,1 }), 1);
    EXPECT_EQ(lr.predict({ 1,0 }), 0);
    EXPECT_EQ(lr.predict({ 2,3 }), 1);
    EXPECT_EQ(lr.predict({ 3,2 }), 0);
    EXPECT_EQ(lr.predict({ 1,1 }), 0);
    EXPECT_EQ(lr.predict({ 1,2 }), 1);
    EXPECT_EQ(lr.predict({ 2,1 }), 0);
}

// Test fit with mismatched X and y sizes
TEST_F(LogisticRegressionTest, FitInvalidSize) {
    std::vector<std::vector<double>> X_data = { {0}, {1}, {2} };
    Matrix X(X_data);
    std::vector<double> y = { 0, 1 };  // Size mismatch

    EXPECT_THROW(lr.fit(X, y), std::invalid_argument);
}

// Test predict with invalid input size
TEST_F(LogisticRegressionTest, PredictInvalidSize) {
    // Fit with valid 1D data
    std::vector<std::vector<double>> X_data = { {0}, {1}, {2} };
    Matrix X(X_data);
    std::vector<double> y = { 0, 0, 1 };
    lr.fit(X, y);

    // Predict with wrong size (2 features instead of 1)
    EXPECT_THROW(lr.predict({ 1,2 }), std::invalid_argument);
}

// Test scalar sigmoid function
TEST_F(LogisticRegressionTest, SigmoidScalar) {
    EXPECT_DOUBLE_EQ(lr.sigmoid(0), 0.5);
    EXPECT_NEAR(lr.sigmoid(100), 1.0, 1e-6);
    EXPECT_NEAR(lr.sigmoid(-100), 0.0, 1e-6);
}

// Test vector sigmoid function
TEST_F(LogisticRegressionTest, SigmoidVector) {
    std::vector<double> z = { 0, 100, -100 };
    std::vector<double> result = lr.sigmoid(z);
    EXPECT_DOUBLE_EQ(result[0], 0.5);
    EXPECT_NEAR(result[1], 1.0, 1e-6);
    EXPECT_NEAR(result[2], 0.0, 1e-6);
}