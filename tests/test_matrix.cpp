#include "matrix.h"
#include <gtest/gtest.h>

// Test matrix constructor
TEST(MatrixTest, Constructor)
{
    Matrix m(2, 3);  // 2 rows, 3 columns
    EXPECT_EQ(m.getNumOfRows(), 2);
    EXPECT_EQ(m.getNumOfCols(), 3);
    EXPECT_DOUBLE_EQ(m(0, 0), 0.0);
}

// Test element access and assignment
TEST(MatrixTest, ElementAccess)
{
    Matrix m(2, 2);
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(1, 0) = 3.0;
    m(1, 1) = 4.0;
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 4.0);
}

// Test matrix transpose
TEST(MatrixTest, Transpose)
{
    Matrix m(2, 3);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
    Matrix mt = m.transpose();
    EXPECT_EQ(mt.getNumOfRows(), 3);
    EXPECT_EQ(mt.getNumOfCols(), 2);
    EXPECT_DOUBLE_EQ(mt(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mt(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(mt(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(mt(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(mt(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(mt(2, 1), 6.0);
}

// Test matrix multiplication
TEST(MatrixTest, MatrixMultiplication)
{
    Matrix m1(2, 3);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0; m1(0, 2) = 3.0;
    m1(1, 0) = 4.0; m1(1, 1) = 5.0; m1(1, 2) = 6.0;
    Matrix m2(3, 2);
    m2(0, 0) = 7.0; m2(0, 1) = 8.0;
    m2(1, 0) = 9.0; m2(1, 1) = 10.0;
    m2(2, 0) = 11.0; m2(2, 1) = 12.0;
    Matrix result = m1 * m2;
    EXPECT_EQ(result.getNumOfRows(), 2);
    EXPECT_EQ(result.getNumOfCols(), 2);
    EXPECT_DOUBLE_EQ(result(0, 0), 58.0);
    EXPECT_DOUBLE_EQ(result(0, 1), 64.0);
    EXPECT_DOUBLE_EQ(result(1, 0), 139.0);
    EXPECT_DOUBLE_EQ(result(1, 1), 154.0);
}

// Test matrix inverse (2x2 case)
TEST(MatrixTest, Inverse)
{
    Matrix m(2, 2);
    m(0, 0) = 4.0; m(0, 1) = 7.0;
    m(1, 0) = 2.0; m(1, 1) = 6.0;
    Matrix inv = m.inverse();
    EXPECT_NEAR(inv(0, 0), 0.6, 1e-10);
    EXPECT_NEAR(inv(0, 1), -0.7, 1e-10);
    EXPECT_NEAR(inv(1, 0), -0.2, 1e-10);
    EXPECT_NEAR(inv(1, 1), 0.4, 1e-10);
}