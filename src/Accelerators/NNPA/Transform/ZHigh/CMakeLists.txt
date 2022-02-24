add_subdirectory(Stickify)

set(LLVM_TARGET_DEFINITIONS ZHighConstPropagation.td)
mlir_tablegen(ZHighConstPropagation.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighConstPropagationIncGen)

add_dlc_library(OMZHighConstPropagation
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
  )

set(LLVM_TARGET_DEFINITIONS ZHighLayoutPropagation.td)
mlir_tablegen(ZHighLayoutPropagation.inc -gen-rewriters "-I${DLC_ONNX_MLIR_SRC_ROOT}" "-I${DLC_SRC_ROOT}")
add_public_tablegen_target(OMZHighLayoutPropagationIncGen)

add_dlc_library(OMZHighLayoutPropagation
  ZHighLayoutPropagation.cpp

  DEPENDS
  OMZHighLayoutPropagationIncGen

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  OMZHighOps
  OMONNXOps
  )

add_dlc_library(OMInstrumentZHigh
  InstrumentZHighPass.cpp

  LINK_LIBS PUBLIC
  OMKrnlOps
  MLIRPass
  OMDLCSupport
  )
