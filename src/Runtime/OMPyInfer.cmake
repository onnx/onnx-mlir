# SPDX-License-Identifier: Apache-2.0

add_subdirectory(python)

add_onnx_mlir_library(OMTensorUtils
  OMInstrument.cpp
  OMTensor.cpp
  OMTensorList.cpp
  OnnxDataType.cpp
  ${ONNX_MLIR_SRC_ROOT}/src/Support/SmallFPConversion.c

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
