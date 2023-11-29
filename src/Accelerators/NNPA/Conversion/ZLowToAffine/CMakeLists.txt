# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMZLowToAffine
  ConvertZLowToAffine.cpp
  ZLowConvertDLF16.cpp

  LINK_LIBS PUBLIC
  OMZLowOps
  MLIRTransforms
  MLIRAffineUtils
  OMONNXToKrnl # this is due to calling insertAllocAndDeallocSimple
  OMKrnlToAffine # this is due to the use of AffineTypeConverter
  )
