# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26.0)

project(torch_onnxmlir_tests NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/utils.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_sessioncache.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_sessioncache.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_add.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_add.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_gemm.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_relu.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_relu.py
  )

file(GLOB PY_TEST_FILES
  "${CMAKE_CURRENT_BINARY_DIR}/test_sessioncache.py"
  "${CMAKE_CURRENT_BINARY_DIR}/test_add.py"
  "${CMAKE_CURRENT_BINARY_DIR}/test_gemm.py"
  "${CMAKE_CURRENT_BINARY_DIR}/test_relu.py"
)

# Use ctest.
enable_testing()

add_custom_target(run-testcases ALL)

foreach(testfile ${PY_TEST_FILES})
    get_filename_component(testname ${testfile} NAME_WE)

    add_test(
        NAME ${testname}
        COMMAND ${Python3_EXECUTABLE} -m pytest -v ${testfile}
    )
endforeach()
