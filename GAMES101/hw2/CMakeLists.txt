cmake_minimum_required(VERSION 3.10)
project(hw2)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
#set(CMAKE_CXX_FLAGS "-O3")  # 开启O3优化

include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include)
include_directories(/usr/local/include)

add_executable(hw2 main.cpp rasterizer.hpp rasterizer.cpp Triangle.hpp Triangle.cpp)
target_link_libraries(hw2 ${OpenCV_LIBS})
