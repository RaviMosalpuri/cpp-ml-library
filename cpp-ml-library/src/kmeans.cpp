#include "kmeans.h"
#include "vector_utils.h"
#include <cmath>
#include <random>

KMeans::KMeans(size_t k, size_t maxIterations, double tolerance)
	:m_k(k), m_maxIterations(maxIterations), m_tolerance(tolerance)
{}

void KMeans::fit(const std::vector<Point>& X)
{
	// Clear the centroids
	m_centroids.clear();
	// Resize the vector
	m_centroids.resize(m_k);
	
	// Randomly assign initial centroids
	std::mt19937 gen(42);
	std::uniform_int_distribution<size_t> dist(0, X.size() - 1);

	// Loop over the centroids
	for (auto& centroid : m_centroids)
	{
		// Assign centroids a random point from input
		centroid = X[dist(gen)];
	}

	// Loop over the iterations
	for (size_t it = 0; it < m_maxIterations; ++it)
	{
		// Initialise clusters
		std::vector<std::vector<Point>> clusters(m_k);

		// Loop over the points
		for (const auto& point : X)
		{
			size_t closestCentroid = getClosestCentroid(point);
			clusters[closestCentroid].emplace_back(point);
		}

		// Initialise new centroids
		std::vector<Point> newCentroids(m_k);

		// Loop over the clusters
		for (size_t i = 0; i < m_k; ++i)
		{
			// Get cluster size
			size_t clusterSize = clusters[i].size();
			
			// Check if size not zero
			if (clusterSize != 0)
			{
				// Initialise sum
				Point sum(X[0].size(), 0.0);
				// Get sum of all points in cluster
				for (const auto& val : clusters[i]) { sum = sum + val; }
				// Get the new centroid as mean of values
				newCentroids[i] = sum / clusterSize;
			}
			else
			{
				// Assign a random value from input
				newCentroids[i] = X[dist(gen)];
			}
		}

		// Store the maximum change in centroid values
		double maxChange = 0.0;
		// Loop over the centroids
		for (size_t i = 0; i < m_k; ++i)
		{
			// Get the maximum change in centroid values
			maxChange = std::max(maxChange, getEuclideanDistance(m_centroids[i], newCentroids[i]));
		}

		// Assign the new centroid values
		m_centroids = std::move(newCentroids);

		// Check if maximum change is less than tolerance value
		if (maxChange < m_tolerance)
		{
			// Break loop
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
	// Initialise the result
	size_t result = 0;

	// Initialise the minimum distance to maximum limit
	double minDist = std::numeric_limits<double>::max();

	// Loop over the centroids
	for (size_t i = 0; i < m_k; ++i)
	{
		// Get the distance from current centroid
		double currDist = getEuclideanDistance(p, m_centroids[i]);
		// Check if current distance is less than minimum distance
		if (currDist < minDist)
		{
			// Update the minimum distance
			minDist = currDist;
			// Update the centroid
			result = i;
		}
	}

	// Return the result
	return result;
}
