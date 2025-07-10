#include "kmeans.h"
#include "vector_utils.h"
#include <cmath>
#include <random>

KMeans::KMeans(size_t k, size_t maxIterations, double tolerance)
	:m_k(k), m_maxIterations(maxIterations), m_tolerance(tolerance)
{}

void KMeans::fit(const std::vector<Point>& X)
{
	m_centroids.clear();
	m_centroids.resize(m_k);
	
	// Randomly assign initial centroids
	std::mt19937 gen(42);
	std::uniform_int_distribution<size_t> dist(0, X.size() - 1);

	for (auto& centroid : m_centroids)
	{
		// Assign centroids a random point from input
		centroid = X[dist(gen)];
	}

	for (size_t it = 0; it < m_maxIterations; ++it)
	{
		// Initialise clusters
		std::vector<std::vector<Point>> clusters(m_k);

		// 1. Assign points to nearest clusters
		for (const auto& point : X)
		{
			size_t closestCentroid = getClosestCentroid(point);
			clusters[closestCentroid].emplace_back(point);
		}

		// 2. Get new centroids as mean of points in clusters
		std::vector<Point> newCentroids(m_k);
		for (size_t i = 0; i < m_k; ++i)
		{
			size_t clusterSize = clusters[i].size();
			
			if (clusterSize != 0)
			{
				Point sum(X[0].size(), 0.0);

				for (const auto& val : clusters[i]) { sum = sum + val; }

				newCentroids[i] = sum / clusterSize;
			}
			else
			{
				// Assign a random value from input
				newCentroids[i] = X[dist(gen)];
			}
		}

		// 3. Calculate change in centroid values

		// Store the maximum change in centroid values
		double maxChange = 0.0;
		for (size_t i = 0; i < m_k; ++i)
		{
			// Get the maximum change in centroid values
			maxChange = std::max(maxChange, getEuclideanDistance(m_centroids[i], newCentroids[i]));
		}

		m_centroids = std::move(newCentroids);

		if (maxChange < m_tolerance)
		{
			break;
		}
	} // for loop
} // fit function

size_t KMeans::predict(const Point& X) const
{
	// Return the closest centroid
	return getClosestCentroid(X);
}

double KMeans::getEuclideanDistance(const Point& a, const Point& b) const
{
	// Return the Euclidean distance between two points
	return std::sqrt(std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2));
}

size_t KMeans::getClosestCentroid(const Point& p) const
{
	size_t result = 0;

	// Initialise the minimum distance to maximum limit
	double minDist = std::numeric_limits<double>::max();

	for (size_t i = 0; i < m_k; ++i)
	{
		double currDist = getEuclideanDistance(p, m_centroids[i]);
		if (currDist < minDist)
		{
			minDist = currDist;
			result = i;
		}
	}

	return result;
}
