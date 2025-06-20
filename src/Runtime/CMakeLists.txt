
# SPDX-License-Identifier: Apache-2.0

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

if (ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
add_compile_definitions(ENABLE_PYRUNTIME_LIGHT)
add_onnx_mlir_library(OMExecutionSession
  ExecutionSession.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMTensorUtils
  )
else()
add_onnx_mlir_library(OMExecutionSession
  ExecutionSession.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
  OMTensorUtils
  LLVMSupport
  )
endif()
set_target_properties(OMExecutionSession
  PROPERTIES
  POSITION_INDEPENDENT_CODE TRUE
  )
