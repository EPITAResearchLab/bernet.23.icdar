cmake_minimum_required(VERSION 3.0.0)
project(AG3LINE)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.h")
include_directories(src/include)
add_library(ag3line ${LIB_SOURCES})
target_link_libraries(ag3line ${OpenCV_LIBS})

add_executable(lsd_ag3line main.cc)
target_link_libraries(lsd_ag3line ag3line ${OpenCV_LIBS})