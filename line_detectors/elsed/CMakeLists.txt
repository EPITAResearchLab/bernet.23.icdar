cmake_minimum_required(VERSION 3.0.0)
project(ELSED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp" "src/*.h")
include_directories(src/include)
add_library(elsed ${LIB_SOURCES})
target_link_libraries(elsed ${OpenCV_LIBS})

add_executable(lsd_elsed main.cc)
target_link_libraries(lsd_elsed elsed ${OpenCV_LIBS})