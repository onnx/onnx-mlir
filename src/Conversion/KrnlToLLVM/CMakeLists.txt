# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToLLVM
  ConvertKrnlToLLVM.cpp
  KrnlFindIndex.cpp
  KrnlEntryPoint.cpp
  KrnlGetRef.cpp  
  KrnlGlobal.cpp
  KrnlInstrument.cpp
  KrnlMemcpy.cpp
  KrnlPrintTensor.cpp
  KrnlPrint.cpp
  KrnlRandomNormal.cpp
  KrnlStrlen.cpp
  KrnlStrncmp.cpp
  KrnlToLLVMHelper.cpp  
  KrnlUnaryMath.cpp
  KrnlVectorTypeCast.cpp 
  RuntimeAPI.cpp

  LINK_LIBS PUBLIC
  Accelerator
  OMSupport
  MLIRAffineToStandard
  MLIRFunc
  MLIRFuncTransforms
  MLIRMathToLLVM
  MLIRMathTransforms
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRShapeToStandard
  MLIRStandardToLLVM
  MLIRVectorToLLVM
  onnx
  )
