add_library(onnx_mlir_transform
        LowerKrnl.cpp
        LowerToLLVM.cpp)

target_include_directories(onnx_mlir_transform
                           PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
                                   ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_transform ${MLIRLibs})
add_dependencies(onnx_mlir_transform gen_krnl_ops)

add_subdirectory(onnx)