add_executable(onnx-mlir-opt onnx_mlir_opt.cpp)
add_dependencies(onnx-mlir-opt gen_krnl_ops)

target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir-opt PRIVATE ${ONNX_MLIR_BIN_ROOT})

target_link_libraries(onnx-mlir-opt
        builder
        ${MLIRLibs}
        onnx_mlir_transform
        onnx_mlir_shape_inference
        onnx_mlir_lower_frontend
        promotable_const_operands_interface
        curses)
whole_archive_link_mlir(onnx-mlir-opt
        ${MLIRWholeArchiveLibs})
whole_archive_link_onnx_mlir(onnx-mlir-opt
        compiler
        onnx_mlir_transform
        onnx_mlir_lower_frontend
        onnx_mlir_shape_inference
        attribute_promotion)