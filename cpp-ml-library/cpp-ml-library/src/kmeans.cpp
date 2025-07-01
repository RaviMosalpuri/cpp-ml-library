#include "kmeans.h"
#include <cmath>
#include <random>

KMeans::KMeans(size_t k, size_t maxIterations, double tolerance)
	:m_k(k), m_maxIterations(maxIterations), m_tolerance(tolerance)
{}

void KMeans::fit(const std::vector<std::vector<double>>& X)
{
	
}

int KMeans::predict(const std::vector<std::vector<double>>& X) const
{
	return 0;
}

double KMeans::getEuclideanDistance(const Point& a, const Point& b)
{
	// Return the Euclidean distance between two points
	return std::sqrt(std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2));
}

size_t KMeans::getClosestCentroid(const Point& p)
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
