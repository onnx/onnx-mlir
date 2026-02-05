
# SPDX-License-Identifier: Apache-2.0

# Light-weight runtime driver supports only python currently
if (NOT ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
add_subdirectory(jni)
add_subdirectory(omp)
endif()
add_subdirectory(python)

# TODO: should add for each accelerator its subdirectory that implements InitAccel##name
# and ShutdownAccel##name.

# Create static libcruntime.a to be embedded in model.so to make model.so self contained.
# However, by default object code for static library is not compiled with -fPIC. Embedding
# such static library in a shared library can cause runtime failure on some architectures,
# such as z. So we override the default and explicitly compile with -fPIC.
add_onnx_mlir_library(cruntime STATIC
  OMExternalConstant.c
  OMIndexLookup.c
  OMInstrument.c
  OMRandomNormal.c
  OMRandomUniform.c
  OMResize.c
  OMSort.c
  OMTopK.c
  OMTensor.c
  OMTensorList.c
  OMUnique.c
  OnnxDataType.c
  ${ONNX_MLIR_SRC_ROOT}/src/Support/SmallFPConversion.c

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
  OMExternalConstant.cpp
  OMIndexLookup.cpp
  OMInstrument.cpp
  OMRandomNormal.cpp
  OMRandomUniform.cpp
  OMResize.cpp
  OMSort.cpp
  OMTopK.cpp
  OMTensor.cpp
  OMTensorList.cpp
  OMUnique.cpp
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
  OMSmallFPConversion
  )
set_target_properties(OMExecutionSession
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )

# For linux, the dynamic library (dl) is used directly, while for Windows,
# LLVMSupport is used for locks when dynamic library is loaded. 
if(WIN32)
  target_link_libraries(OMExecutionSession PRIVATE 
    LLVMSupport
  )
else()
  target_link_libraries(OMExecutionSession PRIVATE dl)
endif()
