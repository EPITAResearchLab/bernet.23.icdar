project(LSDVon_gioi_m)

cmake_minimum_required(VERSION 3.6)

add_library(LSDVon_gioi_m STATIC
    src/lsd.h
    src/lsd.c
    src/main_utils.h
    src/main_utils.c
)

add_executable(lsd_lsd_m main.cc)
target_link_libraries(lsd_lsd_m LSDVon_gioi_m m ${OpenCV_LIBS})