cmake_minimum_required(VERSION 3.15)
project(cpp_ml_library LANGUAGES CXX)

# 1. C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# 3. Build the library
add_library(ml_lib STATIC)
target_sources(ml_lib
    PRIVATE
        src/matrix.cpp
        src/linear_regression.cpp
        src/logistic_regression.cpp
        src/kmeans.cpp
        src/svm.cpp
)
target_include_directories(ml_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 4. Enable testing
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

# Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(mlTests
  tests/test_kmeans.cpp
  tests/test_linear_regression.cpp
  tests/test_logistic_regression.cpp
  tests/test_matrix.cpp
)

target_link_libraries(mlTests
  PRIVATE
  ml_lib
  GTest::gtest
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mlTests)
