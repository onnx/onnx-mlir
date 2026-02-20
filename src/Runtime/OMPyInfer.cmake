# SPDX-License-Identifier: Apache-2.0

# There is subdirectory python for OMPyInfer or not is the same
add_subdirectory(python)

# Assumption here is that CRuntime is included in the .so. We may want to revisit this.
add_onnx_mlir_library(OMTensorUtils
  OMTensor.cpp
  OMTensorList.cpp
  OnnxDataType.cpp
  ${ONNX_MLIR_SRC_ROOT}/src/Support/SmallFPConversion.c

  DEPENDS 
  AcceleratorsInc

  LINK_LIBS PUBLIC

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  )
set_target_properties(OMTensorUtils
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )

add_onnx_mlir_library(OMExecutionSession
  ExecutionSession.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMTensorUtils
  # Needed?
  OMSmallFPConversion
  )
set_target_properties(OMExecutionSession
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )

# For linux, the dynamic library (dl) is used directly, while for Windows,
# LLVMSupport is used for locks when dynamic library is loaded. 
if(WIN32)
  target_link_libraries(OMEexecutionSession PRIVATE 
    LLVMSupport
  )
else()
  target_link_libraries(OMExecutionSession PRIVATE dl)
endif()
