include_directories(include)

add_executable(test_${PROJECT_NAME}
        opencv_unet.cpp)

set(STD_FILESYSTEM)
if (NOT APPLE)
    set(STD_FILESYSTEM stdc++fs)
endif()

target_link_libraries(test_${PROJECT_NAME}
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${STD_FILESYSTEM})

install(TARGETS test_${PROJECT_NAME}
        RUNTIME DESTINATION ${LIBRARY_INSTALL_DESTINATION}/bin)