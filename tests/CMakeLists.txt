enable_testing()

add_executable(mlTests
    test_matrix.cpp
    test_linear_regression.cpp
)

target_link_libraries(mlTests
    PRIVATE
    ml_lib  # Your main library
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mlTests)
