# TODO: remove for release.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "build type")
endif()

# Require 3.3 and set policy CMP0057 for IN_LIST operator support
cmake_minimum_required(VERSION 3.3)
cmake_policy(SET CMP0057 NEW)
project(onnf)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(ONNF_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(ONNF_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#TODO(eventually enable the following)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(third_party/onnx)
add_subdirectory(third_party/benchmark)
add_subdirectory(third_party/pybind11)

# Set cxx standard only for onnf project files.
set(CMAKE_CXX_STANDARD 14)

# Boost 1.61.0 required for boost::dll:program_location()
find_package(Boost 1.61.0
        COMPONENTS graph
        program_options
        log_setup
        log
        system
        filesystem
        REQUIRED)

if(Boost_FOUND)
  message("Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")
  message("Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS}")
  include_directories(${Boost_INCLUDE_DIRS})
endif()

include(MLIR.cmake)
add_subdirectory(src/builder)
add_subdirectory(src/compiler)
add_subdirectory(src)

add_subdirectory(test)

