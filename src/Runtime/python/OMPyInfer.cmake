
# SPDX-License-Identifier: Apache-2.0

# Control source code for float type definition.
add_compile_definitions(ENABLE_PYRUNTIME_LIGHT)
add_onnx_mlir_library(OMPyExecutionSessionBase
  PyExecutionSessionBase.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMExecutionSession
  # ToFix: OMMlirUtilities used for float16 is temporarily excluded because
  # it need some support from LLVM.
  #OMMlirUtilities
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
target_compile_options(PyRuntimeC
  PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:-frtti -fexceptions>
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc /GR>
  )
target_link_libraries(PyRuntimeC
  PRIVATE
  OMPyExecutionSessionBase
  )
llvm_update_compile_flags(PyRuntimeC)

install(TARGETS PyRuntimeC
  DESTINATION lib
  )

# Target to prepare OMPyInfer package
add_custom_target(OMCreateOMPyInferPackage
        COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/OMPyInfer
        COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/OMPyInfer ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND cp ${ONNX_MLIR_BIN_ROOT}/${CMAKE_BUILD_TYPE}/lib/PyRuntimeC.*.so ${CMAKE_CURRENT_BINARY_DIR}/OMPyInfer/src/OMPyInfer/libs
        DEPENDS PyRuntimeC
     )
    
# Target to run OMPyInfer package with newly built PyRuntimeC, not the new
# model.so yet.
add_custom_target(OMTestOMPyInferPackage
        COMMAND pip uninstall -y OMPyInfer
        COMMAND pip install -e ${CMAKE_CURRENT_BINARY_DIR}/OMPyInfer
        COMMAND python ${CMAKE_CURRENT_BINARY_DIR}/OMPyInfer/tests/helloworld.py
        DEPENDS OMCreateOMPyInferPackage
     )
