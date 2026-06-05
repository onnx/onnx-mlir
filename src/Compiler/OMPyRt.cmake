# This library only add MLIR/LLVM independent code that is used both in
# onnx-mlir compiler as well as the OMCompile. The file listed
# here must remain MLIR/LLVM free.

add_onnx_mlir_library(OMCommandUtils
  Command.cpp
  CommandUtils.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  )

add_onnx_mlir_library(OMCompile
  OMCompile.cpp

  EXCLUDE_FROM_OM_LIBS

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PRIVATE
  OMCommandUtils
  )
