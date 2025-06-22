#include "linear_regression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

std::pair<std::vector<double>, std::vector<double>> loadData(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::vector<double> x, y;
    std::string line;
    std::getline(file, line); // Skip header
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ','))
        {
            cells.push_back(cell);
        }
        if (cells.size() != 2) continue;
        x.push_back(std::stod(cells[0]));
        y.push_back(std::stod(cells[1]));
    }
    return { x, y };
}

int main()
{
    try
    {
        auto [x, y] = loadData("data/sample_data.csv");
        for(double v : x) std::cout << v <<std::endl;
        LinearRegression lr(x, y);
        std::cout << "Intercept: " << lr.getIntercept() << std::endl;
        std::cout << "Slope: " << lr.getSlope() << std::endl;
        double pred = lr.predict(5.0);
        std::cout << "Prediction for x=5: " << pred << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}