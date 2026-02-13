# SPDX-License-Identifier: Apache-2.0

# Build cruntime as a shared library and do not link it to model.so in future.
# The libcruntime.so can be put in LD_LIBRARY_PATH or python package libs
# with the PyRuntimeC*.so.
# Verified with zdlc_pyrt test
add_onnx_mlir_library(cruntime SHARED
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

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  )
set_target_properties(cruntime
  PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  )

# Need to link to libm for the sin and cos used in the lib.
# Add more if used in the implementation.
# ToFix: copy needed -l options from the compile option for model?
target_link_libraries(cruntime PRIVATE m)
