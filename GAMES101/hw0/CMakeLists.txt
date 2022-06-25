cmake_minimum_required (VERSION 3.20)
project (hw0)

set(CMAKE_CXX_STANDARD 17)
#find_package(Eigen3 REQUIRED)
#include_directories(EIGEN3_INCLUDE_DIR)

include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include)  # Add Eigen into include path
add_executable (hw0 main.cpp)
