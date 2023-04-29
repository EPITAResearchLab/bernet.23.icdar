project(LSDVon_gioi)

cmake_minimum_required(VERSION 3.6)

add_library(LSDVon_gioi STATIC
    src/lsd.h
    src/lsd.c
    src/main_utils.h
    src/main_utils.c
)

add_executable(lsd_lsd main.cc)
target_link_libraries(lsd_lsd LSDVon_gioi m ${OpenCV_LIBS})