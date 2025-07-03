#include <gtest/gtest.h>
#include "kmeans.h"  // Adjust to your KMeans header file
#include <vector>
#include <cmath>

// Helper function to check if two vectors are approximately equal
bool vectorsApproxEqual(const std::vector<double>& a, const std::vector<double>& b, double tol)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// Test fixture for KMeans
class KMeansTest : public ::testing::Test 
{
protected:
    double tol = 0.1;  // Tolerance for floating-point comparisons
};

// Test basic clustering with two clear clusters
TEST_F(KMeansTest, BasicClustering) 
{
    // Data: two clusters around (0.5, 0.5) and (10.5, 10.5)
    std::vector<std::vector<double>> data = { {0,0}, {1,0}, {0,1}, {1,1}, {10,10}, {11,10}, {10,11}, {11,11} };
    KMeans km(2, 100, 0.001);
    km.fit(data);
    auto centroids = km.getCentroids();

    // Expected centroids: near (0.5, 0.5) and (10.5, 10.5)
    std::vector<double> expected1 = { 0.5, 0.5 };
    std::vector<double> expected2 = { 10.5, 10.5 };

    // Check if centroids are approximately equal to expected values (order agnostic)
    bool c0_matches = vectorsApproxEqual(centroids[0], expected1, tol) || vectorsApproxEqual(centroids[0], expected2, tol);
    bool c1_matches = vectorsApproxEqual(centroids[1], expected1, tol) || vectorsApproxEqual(centroids[1], expected2, tol);
    EXPECT_TRUE(c0_matches && c1_matches);
    EXPECT_FALSE(vectorsApproxEqual(centroids[0], centroids[1], tol));  // Ensure centroids are different

    // Check cluster assignments
    size_t label00 = km.predict({ 0,0 });
    size_t label11 = km.predict({ 1,1 });
    size_t label1010 = km.predict({ 10,10 });
    size_t label1111 = km.predict({ 11,11 });
    EXPECT_EQ(label00, label11);       // Same cluster
    EXPECT_EQ(label1010, label1111);   // Same cluster
    EXPECT_NE(label00, label1010);     // Different clusters
}

// Test with all identical points and K=1
TEST_F(KMeansTest, AllIdenticalPoints)
{
    std::vector<std::vector<double>> data = { {0,0}, {0,0}, {0,0}, {0,0} };
    KMeans km(1, 100, 0.001);
    km.fit(data);
    auto centroids = km.getCentroids();
    EXPECT_NEAR(centroids[0][0], 0.0, 1e-6);
    EXPECT_NEAR(centroids[0][1], 0.0, 1e-6);
    size_t label = km.predict({ 0,0 });
    EXPECT_EQ(label, 0);
}

// Test with K=1 and varied points
TEST_F(KMeansTest, SingleCluster)
{
    std::vector<std::vector<double>> data = { {0,0}, {1,1}, {2,2}, {3,3} };
    KMeans km(1, 100, 0.001);
    km.fit(data);
    auto centroids = km.getCentroids();
    // Expected centroid: mean of points, (1.5, 1.5)
    EXPECT_NEAR(centroids[0][0], 1.5, 1e-6);
    EXPECT_NEAR(centroids[0][1], 1.5, 1e-6);
}

// Test with fewer points than clusters (K=3, 2 points)
TEST_F(KMeansTest, FewerPointsThanClusters)
{
    std::vector<std::vector<double>> data = { {0,0}, {1,1} };
    KMeans km(3, 100, 0.001);
    km.fit(data);
    // Since K=3 but only 2 points, expect two clusters with points and one empty
    auto centroids = km.getCentroids();
    EXPECT_EQ(centroids.size(), 3);  // Still have 3 centroids
    size_t label0 = km.predict({ 0,0 });
    size_t label1 = km.predict({ 1,1 });
    EXPECT_NE(label0, label1);  // Points should be in different clusters
}