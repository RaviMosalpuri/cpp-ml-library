## cpp-ml-library

A modular and extensible machine learning library written in modern C++20, designed for performance, clarity, and educational value. It provides foundational algorithms like linear regression, logistic regression, clustering, and matrix abstractions — implemented from scratch to reinforce core ML concepts.

---

### Key Features

- Linear & Multiple Linear Regression
- Logistic Regression (Batch Gradient Descent)
- K-Means Clustering
- Support Vector Machine (prototype)
- Custom Matrix Class with vectorized operations
- Integrated Google Test Suite via CMake FetchContent
- Installable CMake Package with `find_package(ml)`

---

### Project Structure

```
cpp-ml-library/
├── include/ml/           # Public headers (API)
├── src/                  # ML implementation (.cpp)
├── tests/                # GoogleTest unit tests
├── cmake/                # CMake package configuration templates
├── CMakeLists.txt        # Build + install logic
└── README.md
```

---

### Build & Test Instructions

#### CLI (CMake)

```bash
git clone https://github.com/RaviMosalpuri/cpp-ml-library.git
cd cpp-ml-library

cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install
cmake --build build
cd build && ctest --output-on-failure
```

#### Visual Studio (2022+)

- Open folder → Build `ml_lib` and `mlTests`
- Use Test Explorer to run unit tests
- IDE will auto-discover tests via `gtest_discover_tests()`

---

### GitHub Actions CI
This repository includes a CI workflow `.github/workflows/cpp-ci.yml` that:
- Builds the library using CMake
- Runs tests via ctest
- Validates cross-platform compatibility
- Ensures code quality on every push and pull request

To review or customize the CI pipeline, see: [cpp-ci.yml](.github/workflows/cpp-ci.yml)

---

### Packaging & Installation

To export the library for external use:

```bash
cmake --build build --target install
```

Output layout:

```
install/
├── include/              # ml/*.h headers
├── lib/                  # compiled library (.lib/.a)
└── lib/cmake/ml/         # package metadata
```

#### Use in Another Project

```cmake
find_package(ml REQUIRED PATHS "/absolute/path/to/install/lib/cmake/ml")

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE ml::ml_lib)
```

> Your project will automatically include the correct headers via CMake’s install interface.

---

### Learning Goals

- Build ML tools from scratch using C++ abstractions
- Reinforce numerical methods like gradient descent & clustering
- Practice scalable project architecture and CI-ready build systems
- Apply modern C++ features for clean and expressive code

---

### Planned Additions

- Polynomial Regression
- L1 / L2 Regularization
- Decision Trees, Random Forests
- Naive Bayes Classifier
- Principal Component Analysis (PCA)
- Feedforward Neural Network module
- Optimizer utilities (momentum, Adam)
- Model evaluation metrics (accuracy, log-loss)

---

### License

This project is licensed under the [MIT License](LICENSE).

---
