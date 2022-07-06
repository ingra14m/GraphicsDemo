cmake_minimum_required(VERSION 3.20)
project(hw4)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
include_directories(/usr/local/include)  # include opencv

add_executable(hw4 main.cpp)

target_link_libraries(hw4 ${OpenCV_LIBS})
