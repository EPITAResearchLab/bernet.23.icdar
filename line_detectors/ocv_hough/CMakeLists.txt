project(OCVHough)

add_executable(lsd_ocv_hough main.cc)
target_link_libraries(lsd_ocv_hough ${OpenCV_LIBS})