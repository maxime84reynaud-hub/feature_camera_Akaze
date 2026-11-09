cmake_minimum_required(VERSION 3.16)
project(RobotiqueAKAZE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Charge automatiquement tous les modules OpenCV (compatible OpenCV 4 et OpenCV 5)
find_package(OpenCV REQUIRED)

# Module partagé contenant l'algorithme AKAZE et le prétraitement
add_library(module_features STATIC features.cpp features.h)
target_include_directories(module_features PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(module_features PUBLIC ${OpenCV_LIBS})

# Exécutable 1 : Outil de prétraitement et d'extraction de la carte de référence
add_executable(prep_features main_prep.cpp)
target_link_libraries(prep_features PRIVATE module_features)

# Exécutable 2 : Programme temps réel embarqué (reconnaissance + pose X, Y, Theta)
add_executable(robot_temps_reel main_tracker.cpp)
target_link_libraries(robot_temps_reel PRIVATE module_features)