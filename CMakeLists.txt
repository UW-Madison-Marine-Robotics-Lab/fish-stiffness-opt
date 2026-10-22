cmake_minimum_required(VERSION 3.18)
project(fish_stiffness_opt LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
add_subdirectory(extern/pybind11)

option(FISH_SIM_USE_BLAS "Use a system BLAS library instead of Eigen's built-in backend" OFF)

pybind11_add_module(fish_sim NO_EXTRAS
    bindings.cpp
    src/geometry.cpp
    src/link.cpp
    src/kinematics.cpp
    src/dynamics.cpp
    src/simulation.cpp
)
target_compile_features(fish_sim PRIVATE cxx_std_17)
target_include_directories(fish_sim PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
set_target_properties(fish_sim PROPERTIES
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(FISH_SIM_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_link_libraries(fish_sim PRIVATE BLAS::BLAS)
    target_compile_definitions(fish_sim PRIVATE EIGEN_USE_BLAS)
endif()

install(TARGETS fish_sim LIBRARY DESTINATION . RUNTIME DESTINATION .)
