# SPDX-License-Identifier: Apache-2.0

add_subdirectory(jni)

# TODO: should add for each accelerator its subdirectory that implements InitAccel##name
# and ShutdownAccel##name.

# Create static libcruntime.a to be embedded in model.so to make model.so self contained.
# However, by default object code for static library is not compiled with -fPIC. Embedding
# such static library in a shared library can cause runtime failure on some architectures,
# such as z. So we override the default and explicitly compile with -fPIC.
add_onnx_mlir_library(cruntime STATIC
  OMIndexLookup.c
  OMInstrument.c
  OMRandomNormal.c
  OMResize.c
  OMTensor.c
  OMTensorList.c
  OnnxDataType.c

  DEPENDS
  AcceleratorsInc

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  )
set_target_properties(cruntime
  PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  )

add_onnx_mlir_library(OMTensorUtils
  OMIndexLookup.cpp
  OMInstrument.cpp
  OMRandomNormal.cpp
  OMResize.cpp
  OMTensor.cpp
  OMTensorList.cpp
  OnnxDataType.cpp

  DEPENDS 
  AcceleratorsInc

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  )
set_target_properties(OMTensorUtils
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )

add_onnx_mlir_library(ExecutionSession
  ExecutionSession.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMTensorUtils
  LLVMSupport
  )
set_target_properties(ExecutionSession
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
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
#     (from PyRuntime import ExecutionSession)
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
  ExecutionSession
  )

install(TARGETS PyRuntime
  DESTINATION lib
  )
