cmake_minimum_required(VERSION 3.16)
project(RobotiqueAKAZE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(module_features STATIC features.cpp features.h)
target_include_directories(module_features PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}${OpenCV_INCLUDE_DIRS})
target_link_libraries(module_features PUBLIC ${OpenCV_LIBS})

add_executable(prep_features main_prep.cpp)
target_link_libraries(prep_features PRIVATE module_features)

add_executable(robot_temps_reel main_tracker.cpp)
target_link_libraries(robot_temps_reel PRIVATE module_features)