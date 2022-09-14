cmake_minimum_required(VERSION 3.16.3)
project(eurofilter)

set(CMAKE_BUILD_TYPES "REQUIRED")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# filter
add_subdirectory(${PROJECT_SOURCE_DIR}/filter/)

include_directories(${PROJECT_SOURCE_DIR})
add_executable(eurofilter main.cpp)
target_link_libraries(${PROJECT_NAME} filter ${OpenCV_LIBS} )





