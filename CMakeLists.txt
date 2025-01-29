# SPDX-License-Identifier: Apache-2.0

# Match the minimum required version of LLVM and MLIR
cmake_minimum_required(VERSION 3.20.0)

project(onnx-mlir)

option(ONNX_MLIR_BUILD_TESTS "Build ONNX-MLIR test executables. If OFF, just generate build targets." ON)
option(ONNX_MLIR_CCACHE_BUILD "Set to ON for a ccache enabled build." OFF)
option(ONNX_MLIR_ENABLE_STABLEHLO "Enable StableHLO support." ON)
option(ONNX_MLIR_ENABLE_WERROR "Enable warnings as errors." OFF)
option(ONNX_MLIR_SUPPRESS_THIRD_PARTY_WARNINGS "Suppress warning in third_party code." ON)
option(ONNX_MLIR_ENABLE_JAVA "Set to ON for building the Java runtime, tools, and tests" ON)
option(ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT "Set to ON for building Python driver of running the compiled model without llvm-project." OFF)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# On systems that still have legacy lib64 directories (e.g., rhel/fedora,
# etc.), by default some components (e.g., cmake) install into lib while
# others (e.g., python) install into lib64.
# This causes trouble when we try to figure out the runtime directory in
# CompilerUtils.cpp::getLibraryPath(). So we explicitly set CMAKE_INSTALL_LIBDIR
# to install into lib on all systems so we don't have to deal with lib64.
set(CMAKE_INSTALL_LIBDIR lib)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
# Use the same setting as llvm-project
if (POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# Enable ccache
# This setup is sufficient for Ninja and Makefiles, but not Xcode
if (ONNX_MLIR_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if (CCACHE_PROGRAM)
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
      # RULE_LAUNCH_COMPILE is more powerful than C_COMPILER_LAUNCHER (ccache
      # supports additional compilers) but has issues with cmd and MSVC tools
      set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
      set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    else()
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
    endif()
  else()
    message(FATAL_ERROR "Unable to find the program ccache. Set ONNX_MLIR_CCACHE_BUILD to OFF.")
  endif()
endif()

# Enable warnings as errors
# Leverage the imported LLVM_ENABLE_WERROR for compiler logic
set(LLVM_ENABLE_WERROR ${ONNX_MLIR_ENABLE_WERROR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/bin)
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<CONFIG>/include)

set(ONNX_MLIR_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(ONNX_MLIR_BIN_ROOT ${CMAKE_CURRENT_BINARY_DIR})

set(ONNX_MLIR_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(ONNX_MLIR_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(ONNX_MLIR_INCLUDE_PATH ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

set(ONNX_MLIR_VENDOR ${PACKAGE_VENDOR} CACHE STRING
  "Vendor-specific text for showing with version information.")

if(NOT ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
  include(CTest)
  include(ExternalProject)
endif()
include(MLIR.cmake)

# MLIR.cmake calls find_package(MLIR) which sets LLVM_MINIMUM_PYTHON_VERSION
if (WIN32)
  # On Windows, only search the environment for Python (instead of also checking the registry).
  # This allows us to pick up the correct version of Python instead of always picking the latest
  # installed on the machine.
  set(Python3_FIND_REGISTRY NEVER)
endif()

find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter Development)

if(ONNX_MLIR_ENABLE_JAVA)
  # On MacOS, find_package for Java and JNI doesn't work if JAVA_INCLUDE_PATH
  # and JAVA_INCLUDE_PATH2 are not set. Setting them requires JAVA_HOME to be
  # known. If the JAVA_HOME env var isn't set, we try to figure it out by running
  # java and looking for the java.home property.
  if (APPLE)
    set(JAVA_HOME $ENV{JAVA_HOME})
    if ("${JAVA_HOME}" STREQUAL "")
      execute_process(
        COMMAND bash -c "java -XshowSettings:properties -version 2>&1 | \
                         sed -n 's/[ \t]*java.home = //p'"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE JAVA_HOME)
    endif()
    if (NOT "${JAVA_HOME}" STREQUAL "")
      SET(JAVA_INCLUDE_PATH  "${JAVA_HOME}/include")
      SET(JAVA_INCLUDE_PATH2 "${JAVA_HOME}/include/darwin")
    endif()
  endif()

  # Don't require AWT to allow headless JDK to reduce docker image size
  set(JAVA_AWT_INCLUDE_PATH headless)
  set(JAVA_AWT_LIBRARY headless)
  find_package(Java COMPONENTS Development)
  find_package(JNI)

  if (Java_Development_FOUND AND JNI_FOUND)
    set(ONNX_MLIR_ENABLE_JNI TRUE)
    include(UseJava)
  else()
    set(ONNX_MLIR_ENABLE_JNI FALSE)
    message(WARNING "Java Development component or JNI not found, JNI targets will not work")
  endif()
else()
  set(ONNX_MLIR_ENABLE_JNI FALSE)
  message(WARNING "Java explicitly disabled with the ONNX_MLIR_ENABLE_JAVA option, JNI targets will not work")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # Use the new MSVC preprocessor to improve standard conformance.
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Zc:preprocessor")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")

  # Disable warning for non-standard usages of std::complex template which was
  # introduced in MSVC 14.34.
  # More details at https://github.com/microsoft/STL/pull/2759.
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING")

endif()

# Suppress warnings in third party code if requested.
if (ONNX_MLIR_SUPPRESS_THIRD_PARTY_WARNINGS)
  set(LLVM_ENABLE_WERROR OFF)
  set(CMAKE_C_FLAGS_COPY ${CMAKE_C_FLAGS})
  set(CMAKE_CXX_FLAGS_COPY ${CMAKE_CXX_FLAGS})
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # MSVC complains when overriding existing warning levels flags, so remove them.
    STRING(REGEX REPLACE "/W[0-4xX]" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    STRING(REGEX REPLACE "/W[0-4xX]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS")
  endif()
endif()

# Decrease the log level for the inclusion of third party code
set(CMAKE_MESSAGE_LOG_LEVEL NOTICE)

# Add third party subdirectories and define options appropriate to run their cmakes.
if (ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
  add_subdirectory(third_party/onnx)
  add_subdirectory(third_party/pybind11)
else()
  set(pybind11_FIND_QUIETLY ON)
  add_subdirectory(third_party/onnx)
  add_subdirectory(third_party/pybind11)

  add_subdirectory(third_party/rapidcheck)

  if (ONNX_MLIR_ENABLE_STABLEHLO)
    add_subdirectory(third_party/stablehlo EXCLUDE_FROM_ALL)
  endif()

  if (NOT TARGET benchmark)
    set(BENCHMARK_USE_BUNDLED_GTEST OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_WERROR OFF)
    # Since LLVM requires C++11 (or higher) it is safe to assume that std::regex is available.
    set(HAVE_STD_REGEX ON CACHE BOOL "OK" FORCE)
    add_subdirectory(third_party/benchmark)
  endif()
endif()

# All libraries and executables coming from llvm or ONNX-MLIR have had their
# compile flags updated via llvm_update_compile_flags, so we need to do that to
# benchmark and rapidcheck as well, so that we can successfully link against them.
# Otherwise, some of the flags for exceptions (among others) are not set correctly.
llvm_update_compile_flags(benchmark)
llvm_update_compile_flags(benchmark_main)
llvm_update_compile_flags(rapidcheck)

if (ONNX_MLIR_ENABLE_STABLEHLO)
  llvm_update_compile_flags(stablehlo-opt)
  llvm_update_compile_flags(stablehlo-translate)
endif()

# Increase the log level for onnx-mlir
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)

# Ensure warnings are reported for onnx-mlir code.
if (ONNX_MLIR_SUPPRESS_THIRD_PARTY_WARNINGS)
  set(LLVM_ENABLE_WERROR ${ONNX_MLIR_ENABLE_WERROR})
  set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS_COPY})
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_COPY})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSUPPRESS_THIRD_PARTY_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSUPPRESS_THIRD_PARTY_WARNINGS")
endif()

if (ONNX_MLIR_ENABLE_STABLEHLO)
  add_compile_definitions(ONNX_MLIR_ENABLE_STABLEHLO)
endif()

if (ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
  add_subdirectory(src)
else()
  add_subdirectory(utils)
  add_subdirectory(include)
  add_subdirectory(src)
  add_subdirectory(docs)
  add_subdirectory(test)
endif()
