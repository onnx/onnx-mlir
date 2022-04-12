# SPDX-License-Identifier: Apache-2.0

if (NOT ONNX_MLIR_BUILD_TESTS)
  set(EXCLUDE_FROM_ALL ON)
endif()

add_custom_target(doc-example)
set_target_properties(doc-example PROPERTIES FOLDER "Docs")

add_custom_target(check-doc-example
  COMMENT "Running the documentation example"
  COMMAND "${CMAKE_CTEST_COMMAND}" -L doc-example --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS doc-example
  )

set_target_properties(check-doc-example PROPERTIES FOLDER "Docs")
# Exclude the target from the default VS build
set_target_properties(check-doc-example PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_add_onnx.py
  ${CMAKE_CURRENT_BINARY_DIR}/gen_add_onnx.py
  COPYONLY
  )

if (WIN32)
  set(OMRuntimeTestModelLibraryFiles ${CMAKE_CURRENT_BINARY_DIR}/add${CMAKE_SHARED_LIBRARY_SUFFIX})
  list(APPEND OMRuntimeTestModelLibraryFiles ${CMAKE_CURRENT_BINARY_DIR}/add${CMAKE_LINK_LIBRARY_SUFFIX})
else() 
  set(OMRuntimeTestModelLibraryFiles ${CMAKE_CURRENT_BINARY_DIR}/add${CMAKE_SHARED_MODULE_SUFFIX})
endif()

add_custom_command(
  OUTPUT ${OMRuntimeTestModelLibraryFiles}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/gen_add_onnx.py
  COMMAND onnx-mlir add.onnx
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS onnx-mlir
  )

add_custom_target(OMRuntimeTestModel
  DEPENDS ${OMRuntimeTestModelLibraryFiles}
  )

add_library(OMRuntimeTestModelLibrary SHARED IMPORTED)
add_dependencies(OMRuntimeTestModelLibrary OMRuntimeTestModel)
if (WIN32)
  set_target_properties(OMRuntimeTestModelLibrary PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/add${CMAKE_SHARED_LIBRARY_SUFFIX}
    )
  set_target_properties(OMRuntimeTestModelLibrary PROPERTIES
    IMPORTED_IMPLIB ${CMAKE_CURRENT_BINARY_DIR}/add${CMAKE_LINK_LIBRARY_SUFFIX}
    )
else()
  set_target_properties(OMRuntimeTestModelLibrary PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/add${CMAKE_SHARED_MODULE_SUFFIX}
    )
endif()

add_onnx_mlir_executable(OMRuntimeTest
  main.c

  NO_INSTALL

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  OMRuntimeTestModelLibrary
  cruntime
  )
set_output_directory(OMRuntimeTest BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME OMRuntimeTest COMMAND OMRuntimeTest WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_property(TARGET OMRuntimeTest PROPERTY FOLDER "Docs")
set_tests_properties(OMRuntimeTest PROPERTIES LABELS doc-example)

add_dependencies(doc-example OMRuntimeTest)
