# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToLLVM
  ConvertKrnlToLLVM.cpp
  ConstantOpInterface.cpp
  KrnlFindIndex.cpp
  KrnlCall.cpp
  KrnlEntryPoint.cpp
  KrnlInstrument.cpp
  KrnlMemcpy.cpp
  KrnlNone.cpp
  KrnlPrintTensor.cpp
  KrnlPrint.cpp
  KrnlRandomNormal.cpp
  KrnlRoundEven.cpp
  KrnlStrlen.cpp
  KrnlStrncmp.cpp
  KrnlToLLVMHelper.cpp  
  KrnlUnaryMath.cpp
  KrnlVectorTypeCast.cpp 
  RuntimeAPI.cpp

  LINK_LIBS PUBLIC
  OMAccelerator
  OMConstantOpInterface
  OMSupport
  MLIRAffineToStandard
  MLIRArithTransforms
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRFuncTransforms
  MLIRMathToLLVM
  MLIRMathTransforms
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIROpenMPToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRShapeToStandard
  MLIRVectorToLLVMPass
  onnx
  )
