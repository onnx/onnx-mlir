# SPDX-License-Identifier: Apache-2.0

# TODO: Remove pybind11::python_link_helper after cmake version bumped to 3.16+
add_onnx_mlir_library(OMPyExecutionSessionBase
  PyExecutionSessionBase.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMExecutionSession
  OMMlirUtilities
  pybind11::embed
  pybind11::python_link_helper
  )
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
#   - PyRuntime.cpython-39-s390x-linux-gnu.so
#     (from PyRuntime import OMExecutionSession)
#
# Both libs share the same libprotobuf.so when loaded by test.py.
# However, they were both built with the same onnx-ml.pb.cc generated
# from onnx-ml.proto and the protobuf runtime requires all compiled-in
# .proto files have unique names. Hence the error.
#
# PyRuntime doesn't really need onnx beyond the onnx::TensorProto::*
# types so we remove onnx from its target_link_libraries. But that
# also removes some of the compile definitions and include directories
# which we add back through target_compile_definitions and
# target_include_directories.
pybind11_add_module(PyRuntime PyExecutionSession.cpp)
add_dependencies(PyRuntime onnx_proto)
target_compile_options(PyRuntime
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-frtti -fexceptions>
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
  )
target_compile_definitions(PyRuntime
  PRIVATE
  $<TARGET_PROPERTY:onnx,COMPILE_DEFINITIONS>
  )
target_include_directories(PyRuntime
  PRIVATE
  $<TARGET_PROPERTY:onnx,INCLUDE_DIRECTORIES>
  )
target_link_libraries(PyRuntime
  PRIVATE
  OMPyExecutionSessionBase
  )
llvm_update_compile_flags(PyRuntime)

install(TARGETS PyRuntime
  DESTINATION lib
  )

pybind11_add_module(PyCompileAndRuntime PyOMCompileExecutionSession.cpp)
add_dependencies(PyCompileAndRuntime onnx_proto)
target_compile_options(PyCompileAndRuntime
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-frtti -fexceptions>
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
  )
target_compile_definitions(PyCompileAndRuntime
  PRIVATE
  $<TARGET_PROPERTY:onnx,COMPILE_DEFINITIONS>
  )
target_include_directories(PyCompileAndRuntime
  PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  $<TARGET_PROPERTY:onnx,INCLUDE_DIRECTORIES>
  )
target_link_libraries(PyCompileAndRuntime
  PRIVATE
  OMCompiler
  OMPyExecutionSessionBase
  )
llvm_update_compile_flags(PyCompileAndRuntime)

install(TARGETS PyCompileAndRuntime
  DESTINATION lib
  )
