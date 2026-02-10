# SPDX-License-Identifier: Apache-2.0

function(add_pass_group NAME TD_FILE PREFIX)
  set(LLVM_TARGET_DEFINITIONS ${TD_FILE}.td)
  mlir_tablegen(${TD_FILE}.h.inc -gen-pass-decls -name ${NAME})
  mlir_tablegen(${PREFIX}.capi.h.inc -gen-pass-capi-header --prefix ${PREFIX})
  mlir_tablegen(${PREFIX}.capi.cpp.inc -gen-pass-capi-impl --prefix ${PREFIX})
  add_public_tablegen_target(OM${NAME}PassIncGen)
endfunction()

#separate krnl pass into different .td
add_pass_group(TransformsKrnl PassesKrnl TransformsKrnl)
add_pass_group(Transforms Passes Transforms)

add_onnx_mlir_library(OMLowerKrnlRegion
  LowerKrnlRegion.cpp

  DEPENDS
  OMTransformsKrnlPassIncGen

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMScfParallelPrivateRegion
  ProcessScfParallelPrivate.cpp
  ProcessKrnlParallelClause.cpp

  DEPENDS
  OMTransformsPassIncGen

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  MLIROpenMPToLLVM
  )

add_onnx_mlir_library(OMInstrument
  InstrumentPass.cpp
  InstrumentCleanupPass.cpp

  DEPENDS
  OMTransformsPassIncGen

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMONNXOps
  OMKrnlOps
  MLIRPass
  OMOptionUtils
  )

add_onnx_mlir_library(BufferOMPLoopHoist
  BufferOMPLoopHoist.cpp

  DEPENDS
  OMTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIROpenMPDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  OMOptionUtils
  )

add_onnx_mlir_library(OMSetDiscardableAttr
  SetDiscardableAttr.cpp

  DEPENDS
  OMTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  OMSupport
  )
