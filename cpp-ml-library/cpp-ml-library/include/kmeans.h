#ifndef KMEANS_H
#define KMEANS_H

#include <memory>
#include <vector>

using Point = std::vector<double>;

/// <summary>
/// Class for implementation of K Means algorithm
/// </summary>
class KMeans
{
public:
	/// <summary>
	/// Constructor
	/// </summary>
	/// <param name="k">Number of clusters</param>
	/// <param name="maxIterations">Maximum number of iterations</param>
	/// <param name="tolerance">Minimum tolerance value</param>
	KMeans(size_t k, size_t maxIterations = 100, double tolerance = 0.001);

	/// <summary>
	/// Fit the data
	/// </summary>
	/// <param name="X">Input data</param>
	void fit(const std::vector<Point>& X);

	/// <summary>
	/// Predict
	/// </summary>
	/// <param name="X">Input data</param>
	/// <returns>Predicted cluster value</returns>
	size_t predict(const Point& X) const;

	/// <summary>
	/// Get the centroids
	/// </summary>
	/// <returns>Centroids</returns>
	std::vector<Point> getCentroids() const { return m_centroids; }
private:

	/// <summary>
	/// Calculated euclidean distance between two points
	/// </summary>
	/// <param name="a">Point a</param>
	/// <param name="b">Point a</param>
	/// <returns>Euclidean distance</returns>
	double getEuclideanDistance(const Point& a, const Point& b) const;

	/// <summary>
	/// Calculates the closest centroid to a given point
	/// </summary>
	/// <param name="p">Input point</param>
	/// <returns>Closest centroid</returns>
	size_t getClosestCentroid(const Point& p) const;

	// Number of clusters
	size_t m_k;

	// Maximum number of iterations
	size_t m_maxIterations;

	// Minimum tolerance value
	double m_tolerance;

	// Centroids
	std::vector<Point> m_centroids;
};

#endif // !KMEANS_H
