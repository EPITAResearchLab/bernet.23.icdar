project(pylene-line-detector)

find_package(pylene REQUIRED)

include_directories(${pylene_INCLUDE_DIR})
add_executable(lsd_pylene main.cc)
target_link_libraries(lsd_pylene pylene::pylene ${OpenCV_LIBS})