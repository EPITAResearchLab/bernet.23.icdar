cmake_minimum_required(VERSION 3.6)

project(ED_Lib)

set(CMAKE_CXX_STANDARD 11)

add_library(EDLib STATIC
        src/ED.cpp
        src/ED.h
        src/EDCircles.cpp
        src/EDCircles.h
        src/EDColor.cpp
        src/EDColor.h
        src/EDLib.h
        src/EDLines.cpp
        src/EDLines.h
        src/EDPF.cpp
        src/EDPF.h
        src/NFA.cpp
        src/NFA.h)

add_executable(lsd_edlines main.cc)
target_link_libraries(lsd_edlines ${OpenCV_LIBS} EDLib)