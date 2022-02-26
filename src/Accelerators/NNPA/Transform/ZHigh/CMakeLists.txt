add_subdirectory(Stickify)

set(LLVM_TARGET_DEFINITIONS ZHighConstPropagation.td)
mlir_tablegen(ZHighConstPropagation.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighConstPropagationIncGen)

add_onnx_mlir_library(OMZHighConstPropagation
  ZHighConstPropagation.cpp

  DEPENDS
  OMZHighConstPropagationIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMLayoutHelper
  OMStickify
  OMZHighOps
  OMONNXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  )

set(LLVM_TARGET_DEFINITIONS ZHighLayoutPropagation.td)
mlir_tablegen(ZHighLayoutPropagation.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighLayoutPropagationIncGen)

add_onnx_mlir_library(OMZHighLayoutPropagation
  ZHighLayoutPropagation.cpp

  DEPENDS
  OMZHighLayoutPropagationIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZHighOps
  OMONNXOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
   )

add_onnx_mlir_library(OMInstrumentZHigh
  InstrumentZHighPass.cpp

  LINK_LIBS PUBLIC
  OMKrnlOps
  MLIRPass
  OMDLCSupport

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
   )
