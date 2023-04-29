PROJECT( CannyLine )

add_library(CannyLib STATIC
    src/buffer_2d.cpp
    src/buffer_2d.h
    src/CannyLine.cpp
    src/CannyLine.h
    src/MetaLine.cpp       
    src/MetaLine.h
    src/QuickSort.h
    src/types.h
    )

add_executable(lsd_cannylines main.cc)
target_link_libraries(lsd_cannylines ${OpenCV_LIBS} CannyLib)