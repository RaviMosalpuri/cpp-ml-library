# ifndef VECTOR_UTILS_H
# define VECTOR_UTILS_H

#include <vector>
#include <stdexcept>

/// <summary>
/// Addition operator for subtracting two vectors
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="vec1">Vector 1</param>
/// <param name="vec2">Vector 2</param>
/// <returns>Resultant vector</returns>
template <typename T>
std::vector<T> operator+(const std::vector<T>& vec1, const std::vector<T>& vec2)
{
    // Check if vectors have same size
    if (vec1.size() != vec2.size())
    {
        // Throw an invalid argument exception
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    // Get size of vector
    size_t size = vec1.size();

    // Intialise resultant vector
    std::vector<T> result(size);

    // Loop over elements
    for (size_t i = 0; i < size; ++i)
    {
        // Get the difference of elements
        result[i] = vec1[i] + vec2[i];
    }

    // Return result
    return result;
}

/// <summary>
/// Minus operator for subtracting two vectors
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="vec1">Vector 1</param>
/// <param name="vec2">Vector 2</param>
/// <returns>Resultant vector</returns>
template <typename T>
std::vector<T> operator-(const std::vector<T>& vec1, const std::vector<T>& vec2)
{
    // Check if vectors have same size
    if (vec1.size() != vec2.size())
    {
        // Throw an invalid argument exception
        throw std::invalid_argument("Vectors must be of the same size.");
    }

    // Get size of vector
    size_t size = vec1.size();

    // Intialise resultant vector
    std::vector<T> result(size);

    // Loop over elements
    for (size_t i = 0; i < size; ++i)
    {
        // Get the difference of elements
        result[i] = vec1[i] - vec2[i];
    }

    // Return result
    return result;
}

/// <summary>
/// Division operator for dividing vector using a value
/// </summary>
/// <typeparam name="T"></typeparam>
/// <typeparam name="S"></typeparam>
/// <param name="vec">Input vecctor</param>
/// <param name="val">Input value</param>
/// <returns>Resultant vector</returns>
template <typename T, typename S>
std::vector<T> operator/(const std::vector<T>& vec, S val)
{
    // Check if vector is empty
    if (vec.empty())
    {
        // Throw an invalid argument exception
        throw std::invalid_argument("Vector must not be empty.");
    }
    // Check if value is zero
    else if (val == 0)
    {
        // Throw an invalid argument exception
        throw std::invalid_argument("Cannot divide by zero.");
    }

    // Get size of vector
    size_t size = vec.size();

    // Intialise resultant vector
    std::vector<T> result(size);

    // Loop over elements
    for (size_t i = 0; i < size; ++i)
    {
        // Divide each element by value
        result[i] = vec[i]/val;
    }

    // Return the result
    return result;
}

/// <summary>
/// Multiplication operator for multiplying vector by a value
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="vec">Input vector</param>
/// <param name="val">Input value</param>
/// <returns>Resultant vector</returns>
template<typename T>
std::vector<T> operator*(const std::vector<T> vec, T val)
{
    // Check if vector is empty
    if (vec.empty())
    {
        // Throw an invalid argument exception
        throw std::invalid_argument("Vector must not be empty.");
    }

    // Get size of vector
    size_t size = vec.size();

    // Intialise resultant vector
    std::vector<T> result(size);

    // Loop over elements
    for (size_t i = 0; i < size; ++i)
    {
        // Multiply each element by value
        result[i] = val * vec[i];
    }

    // Return the result
    return result;
}

/// <summary>
/// Multiplication operator for multiplying vector by a value
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="val">Input value</param>
/// <param name="vec">Input vector</param>
/// <returns>Resultant vector</returns>
template<typename T>
std::vector<T> operator*(T val, const std::vector<T> vec)
{
    // Call the other multiplication operator and return the vector
    return vec * val;
}

#endif // !VECTOR_UTILS_H
