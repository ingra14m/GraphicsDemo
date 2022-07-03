cmake_minimum_required(VERSION 3.20)
project(hw3)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include)   # include eigen
include_directories(/usr/local/include)  # include opencv

add_executable(hw3 main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h)
target_link_libraries(hw3 ${OpenCV_LIBS})
#target_compile_options(Rasterizer PUBLIC -Wall -Wextra -pedantic)