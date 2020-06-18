add_executable(onnx-mlir-opt ONNXMLIROpt.cpp)
add_dependencies(onnx-mlir-opt OMKrnlOpsInc OMONNXOpsInc)

target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_BIN_ROOT})

set(ONNX_MLIR_LD_PRELOAD_onnx-mlir-opt "" CACHE STRING "" FORCE)
whole_archive_link_onnx_mlir(onnx-mlir-opt ${ONNXMLIRWholeArchiveLibs})
whole_archive_link_mlir(onnx-mlir-opt ${MLIRWholeArchiveLibs})
if(BUILD_SHARED_LIBS)
  message(STATUS "To run dynamically linked onnx-mlir-opt, you must specify:")
  message(STATUS "LD_PRELOAD=${ONNX_MLIR_LD_PRELOAD_onnx-mlir-opt}")
endif()

target_link_libraries(onnx-mlir-opt
        OMBuilder
        OMKrnlOps
        OMONNXOps
        OMONNXRewrite
        ${MLIRLibs})

if(BUILD_SHARED_LIBS)
  target_link_libraries(onnx-mlir-opt SHARED
          ${ONNXMLIRWholeArchiveLibs})
  target_link_libraries(onnx-mlir-opt SHARED
          ${MLIRWholeArchiveSharedLibs})
endif()
