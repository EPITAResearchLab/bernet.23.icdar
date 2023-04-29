project(line-segment-detectors)

cmake_minimum_required(VERSION 3.11)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/../bin)
if (NOT EXISTS "${EXECUTABLE_OUTPUT_PATH}")
    file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
endif()

set(CMAKE_CXX_STANDARD 20)

# Used in subprojects
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(pylene-ours)

add_subdirectory(ag3line)
add_subdirectory(edlib)
add_subdirectory(ocv_hough)
add_subdirectory(cannylines)
add_subdirectory(elsed)
add_subdirectory(lsd_1.6)
add_subdirectory(lsd_1.6_m)