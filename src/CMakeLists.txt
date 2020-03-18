add_library(compiler
        Dialect/Krnl/KrnlOps.cpp
        Dialect/Krnl/KrnlOps.hpp
        Dialect/Krnl/KrnlTypes.cpp
        Dialect/Krnl/KrnlTypes.hpp
        Dialect/ONNX/ONNXOps.cpp
        Dialect/ONNX/ONNXOps.hpp
        Dialect/Krnl/KrnlHelper.cpp
        Dialect/Krnl/KrnlHelper.hpp
        Pass/ShapeInferenceInterface.hpp
        Pass/ONNXCombine.cpp
        Pass/ONNXRewrite.cpp
        Pass/ONNXDecompose.cpp
        Pass/Passes.hpp)

# Include root src directory.
target_include_directories(compiler PRIVATE ${ONNX_MLIR_SRC_ROOT})

# Include tablegen generated header files.
target_include_directories(compiler PRIVATE ${ONNX_MLIR_BIN_ROOT})

target_link_libraries(compiler
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        ${MLIRLibs}
        curses)

set(LLVM_TARGET_DEFINITIONS Pass/ONNXDecompose.td)
onnx_mlir_tablegen(onnx_decompose.inc -gen-rewriters)
add_public_tablegen_target(gen_onnx_decompose)
add_dependencies(compiler gen_onnx_decompose)

set(LLVM_TARGET_DEFINITIONS Pass/ShapeInferenceInterface.td)
onnx_mlir_tablegen(shape_inference.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(shape_inference.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(gen_shape_inference)
add_dependencies(compiler gen_shape_inference)

set(LLVM_TARGET_DEFINITIONS Pass/ONNXCombine.td)
onnx_mlir_tablegen(onnx_combine.inc -gen-rewriters)
add_public_tablegen_target(gen_onnx_combine)
add_dependencies(compiler gen_onnx_combine)

set(LLVM_TARGET_DEFINITIONS Pass/ONNXRewrite.td)
onnx_mlir_tablegen(onnx_rewrite.inc -gen-rewriters)
add_public_tablegen_target(gen_onnx_rewrite)
add_dependencies(compiler gen_onnx_rewrite)

set(LLVM_TARGET_DEFINITIONS Dialect/ONNX/ONNXOps.td)
onnx_mlir_tablegen(onnx.hpp.inc -gen-op-decls "-I${CMAKE_SOURCE_DIR}/compiler/pass")
onnx_mlir_tablegen(onnx.cpp.inc -gen-op-defs "-I${CMAKE_SOURCE_DIR}/compiler/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/onnx.md)
add_public_tablegen_target(gen_onnx)
add_dependencies(compiler gen_onnx)
add_onnx_mlir_dialect_doc(onnx Dialect/ONNX/ONNXOps.td)

set(LLVM_TARGET_DEFINITIONS Dialect/Krnl/KrnlOps.td)
onnx_mlir_tablegen(krnl.hpp.inc -gen-op-decls)
onnx_mlir_tablegen(krnl.cpp.inc -gen-op-defs)
add_public_tablegen_target(gen_krnl_ops)
add_dependencies(compiler gen_krnl_ops)

add_library(onnx_mlir_onnx_decompose Pass/ONNXDecompose.cpp)
target_include_directories(onnx_mlir_onnx_decompose
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_onnx_decompose ${MLIRLibs})
add_dependencies(onnx_mlir_onnx_decompose gen_krnl_ops)

add_library(onnx_mlir_shape_inference Pass/ShapeInferencePass.cpp)
target_include_directories(onnx_mlir_shape_inference
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_shape_inference ${MLIRLibs})
add_dependencies(onnx_mlir_shape_inference gen_krnl_ops)

add_library(onnx_mlir_lower_frontend
        Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp
        Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp
        Conversion/ONNXToKrnl/Math/Elementwise.cpp
        Conversion/ONNXToKrnl/Math/Gemm.cpp
        Conversion/ONNXToKrnl/Math/MatMul.cpp
        Conversion/ONNXToKrnl/Math/Reduction.cpp
        Conversion/ONNXToKrnl/Math/Softmax.cpp
        Conversion/ONNXToKrnl/NN/Conv.cpp
        Conversion/ONNXToKrnl/NN/Normalization.cpp
        Conversion/ONNXToKrnl/NN/Pooling.cpp
        Conversion/ONNXToKrnl/Tensor/Identity.cpp
        Conversion/ONNXToKrnl/Tensor/Reshape.cpp
        Conversion/ONNXToKrnl/Tensor/PadConstantValuePad.cpp
        Conversion/ONNXToKrnl/Tensor/Transpose.cpp
        Conversion/ONNXToKrnl/Tensor/Unsqueeze.cpp
        Conversion/ONNXToKrnl/Tensor/Constant.cpp
        Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp)
target_include_directories(onnx_mlir_lower_frontend
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_lower_frontend ${MLIRLibs})
add_dependencies(onnx_mlir_lower_frontend gen_krnl_ops)

add_subdirectory(transform)
add_subdirectory(Tool)
add_subdirectory(Builder)
add_subdirectory(Runtime)

add_executable(onnx-mlir main.cpp)

target_link_libraries(onnx-mlir
        builder
        ${MLIRLibs}
        onnx_mlir_transform
        onnx_mlir_onnx_decompose
        onnx_mlir_shape_inference
        onnx_mlir_lower_frontend)
whole_archive_link_mlir(onnx-mlir ${MLIRWholeArchiveLibs})
find_package(ZLIB REQUIRED)
target_link_libraries(onnx-mlir ${ZLIB_LIBRARIES})

target_include_directories(onnx-mlir PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS onnx-mlir DESTINATION bin)
