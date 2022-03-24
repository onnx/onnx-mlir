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
  OMSupport
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRMathTransforms
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRShapeToStandard
  MLIRStandardOpsTransforms
  MLIRStandardToLLVM
  MLIRVectorToLLVM
  onnx
  )

# On Windows, we need an exports file which contains the names of each function expected in the model library
if (WIN32)
  add_custom_target(OMModelLibraryExportFile
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/OnnxMlirExports.def ${ONNX_MLIR_LIBRARY_PATH}
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/OnnxMlirExports.def
    )
  add_dependencies(OMKrnlToLLVM OMModelLibraryExportFile)
  install(FILES OnnxMlirExports.def DESTINATION lib)
endif()
