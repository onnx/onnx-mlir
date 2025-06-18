# SPDX-License-Identifier: Apache-2.0

file(GENERATE
  OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/PyRuntime.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/PyRuntime.py
  )

file(GENERATE
  OUTPUT ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/PyCompileAndRuntime.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/PyCompileAndRuntime.py
  )

# TODO: Remove pybind11::python_link_helper after cmake version bumped to 3.16+
if(NOT ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
  add_onnx_mlir_library(OMPyExecutionSessionBase
    PyExecutionSessionBase.cpp

    EXCLUDE_FROM_OM_LIBS

    LINK_LIBS PUBLIC
    OMExecutionSession
    OMMlirUtilities
    pybind11::embed
    pybind11::python_link_helper
    )
else()
  add_compile_definitions(ENABLE_PYRUNTIME_LIGHT)
  add_onnx_mlir_library(OMPyExecutionSessionBase
    PyExecutionSessionBase.cpp

    EXCLUDE_FROM_OM_LIBS

    LINK_LIBS PUBLIC
    OMExecutionSession
    pybind11::embed
    pybind11::python_link_helper
  )
endif()

if(MSVC)
  target_link_libraries(OMPyExecutionSessionBase
    PRIVATE pybind11::windows_extras
  )
endif()
set_target_properties(OMPyExecutionSessionBase
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )
target_compile_definitions(OMPyExecutionSessionBase
  PRIVATE
  $<TARGET_PROPERTY:onnx,COMPILE_DEFINITIONS>
  )
target_include_directories(OMPyExecutionSessionBase
  PRIVATE
  $<TARGET_PROPERTY:onnx,INCLUDE_DIRECTORIES>
  )

# When running on ubi8 image, shared lib backend tests fail with
# the following error:
#
#    [libprotobuf ERROR google/protobuf/descriptor_database.cc:641] File already exists in database: onnx/onnx-ml.proto
#    [libprotobuf FATAL google/protobuf/descriptor.cc:1371] CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):
#    terminate called after throwing an instance of 'google::protobuf::FatalException'
#      what():  CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size):
#    Aborted (core dumped)
#
# This is because test.py loads (among others) the following
# two .so shared libs:
#
#   - onnx_cpp2py_export.cpython-39-s390x-linux-gnu.so
#     (import onnx)
#   - PyRuntimeC.cpython-39-s390x-linux-gnu.so
#     (from PyRuntimeC import OMExecutionSession)
#
# Both libs share the same libprotobuf.so when loaded by test.py.
# However, they were both built with the same onnx-ml.pb.cc generated
# from onnx-ml.proto and the protobuf runtime requires all compiled-in
# .proto files have unique names. Hence the error.
#
# PyRuntimeC doesn't really need onnx beyond the onnx::TensorProto::*
# types so we remove onnx from its target_link_libraries. But that
# also removes some of the compile definitions and include directories
# which we add back through target_compile_definitions and
# target_include_directories.
pybind11_add_module(PyRuntimeC PyExecutionSession.cpp)
add_dependencies(PyRuntimeC onnx_proto)
target_compile_options(PyRuntimeC
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-frtti -fexceptions>
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
  )
target_compile_definitions(PyRuntimeC
  PRIVATE
  $<TARGET_PROPERTY:onnx,COMPILE_DEFINITIONS>
  )
target_include_directories(PyRuntimeC
  PRIVATE
  $<TARGET_PROPERTY:onnx,INCLUDE_DIRECTORIES>
  )
target_link_libraries(PyRuntimeC
  PRIVATE
  OMPyExecutionSessionBase
  )
llvm_update_compile_flags(PyRuntimeC)

install(TARGETS PyRuntimeC
  DESTINATION lib
  )

# Target to prepare onnxmlir package
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onnxmlir/src/onnxmlir/libs)

add_custom_target(OMCreatePyRuntimePackage
        COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/onnxmlir ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/onnxmlirdocker.py ${CMAKE_CURRENT_BINARY_DIR}/onnxmlir/src/onnxmlir/
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/PyRuntime.py ${CMAKE_CURRENT_BINARY_DIR}/onnxmlir/src/onnxmlir
        COMMAND cp ${ONNX_MLIR_BIN_ROOT}/${CMAKE_BUILD_TYPE}/lib/PyRuntimeC.*.so ${CMAKE_CURRENT_BINARY_DIR}/onnxmlir/src/onnxmlir/libs
        DEPENDS PyRuntimeC
     )

# Target to prepare onnxmlirtorch package
add_custom_target(OMCreateONNXMLIRTorchPackage
        COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/onnxmlirtorch ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND cp ${ONNX_MLIR_BIN_ROOT}/${CMAKE_BUILD_TYPE}/lib/PyRuntimeC.*.so ${CMAKE_CURRENT_BINARY_DIR}/onnxmlirtorch/src/onnxmlirtorch/
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/onnxmlirdocker.py ${CMAKE_CURRENT_BINARY_DIR}/onnxmlirtorch/src/onnxmlirtorch
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/PyRuntime.py ${CMAKE_CURRENT_BINARY_DIR}/onnxmlirtorch/src/onnxmlirtorch
        DEPENDS PyRuntimeC
     )

if(ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
  # Since the compiler is not built when PYRUNTIME_LIGHT is enabled,
  # PyCompileAndRuntime can not be simply built.
  return()
endif()

pybind11_add_module(PyCompileAndRuntimeC PyOMCompileExecutionSession.cpp)
add_dependencies(PyCompileAndRuntimeC onnx_proto)
target_compile_options(PyCompileAndRuntimeC
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-frtti -fexceptions>
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
  )
target_compile_definitions(PyCompileAndRuntimeC
  PRIVATE
  $<TARGET_PROPERTY:onnx,COMPILE_DEFINITIONS>
  )
target_include_directories(PyCompileAndRuntimeC
  PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  $<TARGET_PROPERTY:onnx,INCLUDE_DIRECTORIES>
  )
target_link_libraries(PyCompileAndRuntimeC
  PRIVATE
  OMCompiler
  OMPyExecutionSessionBase
  )
llvm_update_compile_flags(PyCompileAndRuntimeC)

install(TARGETS PyCompileAndRuntimeC
  DESTINATION lib
  )

