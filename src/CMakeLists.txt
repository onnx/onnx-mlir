add_library(compiler
        dialect/krnl/krnl_ops.cpp
        dialect/krnl/krnl_ops.hpp
        dialect/krnl/krnl_types.cpp
        dialect/krnl/krnl_types.hpp
        dialect/onnx/onnx_ops.cpp
        dialect/onnx/onnx_ops.hpp
        dialect/krnl/krnl_helper.cpp
        dialect/krnl/krnl_helper.hpp
        pass/shape_inference_interface.hpp
        pass/onnx_combine.cpp
        pass/onnx_rewrite.cpp
        pass/onnx_decompose.cpp
        pass/passes.hpp)

# Include root src directory.
target_include_directories(compiler PRIVATE ${ONNX_MLIR_SRC_ROOT})

# Include tablegen generated header files.
target_include_directories(compiler PRIVATE ${ONNX_MLIR_BIN_ROOT})

target_link_libraries(compiler
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        ${MLIRLibs}
        curses)

set(LLVM_TARGET_DEFINITIONS pass/onnx_decompose.td)
onnx_mlir_tablegen(onnx_decompose.inc -gen-rewriters)
add_public_tablegen_target(gen_onnx_decompose)
add_dependencies(compiler gen_onnx_decompose)

set(LLVM_TARGET_DEFINITIONS pass/shape_inference_interface.td)
onnx_mlir_tablegen(shape_inference.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(shape_inference.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(gen_shape_inference)
add_dependencies(compiler gen_shape_inference)

set(LLVM_TARGET_DEFINITIONS pass/onnx_combine.td)
onnx_mlir_tablegen(onnx_combine.inc -gen-rewriters)
add_public_tablegen_target(gen_onnx_combine)
add_dependencies(compiler gen_onnx_combine)

set(LLVM_TARGET_DEFINITIONS pass/onnx_rewrite.td)
onnx_mlir_tablegen(onnx_rewrite.inc -gen-rewriters)
add_public_tablegen_target(gen_onnx_rewrite)
add_dependencies(compiler gen_onnx_rewrite)

add_subdirectory(interface)

set(LLVM_TARGET_DEFINITIONS dialect/onnx/onnx.td)
onnx_mlir_tablegen(onnx.hpp.inc -gen-op-decls "-I${CMAKE_SOURCE_DIR}/compiler/pass")
onnx_mlir_tablegen(onnx.cpp.inc -gen-op-defs "-I${CMAKE_SOURCE_DIR}/compiler/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/onnx.md)
add_public_tablegen_target(gen_onnx)

add_dependencies(gen_onnx gen_shape_inference)
add_dependencies(compiler gen_onnx)

add_onnx_mlir_dialect_doc(onnx dialect/onnx/onnx.td)

set(LLVM_TARGET_DEFINITIONS dialect/krnl/krnl_ops.td)
onnx_mlir_tablegen(krnl.hpp.inc -gen-op-decls)
onnx_mlir_tablegen(krnl.cpp.inc -gen-op-defs)
add_public_tablegen_target(gen_krnl_ops)
add_dependencies(compiler gen_krnl_ops)

add_library(onnx_mlir_onnx_decompose pass/onnx_decompose.cpp)
target_include_directories(onnx_mlir_onnx_decompose
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_onnx_decompose ${MLIRLibs})
add_dependencies(onnx_mlir_onnx_decompose gen_onnx)

add_library(onnx_mlir_shape_inference pass/shape_inference_pass.cpp)
target_include_directories(onnx_mlir_shape_inference
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_shape_inference ${MLIRLibs})
add_dependencies(onnx_mlir_shape_inference gen_onnx)

add_library(onnx_mlir_lower_frontend
        conversion/onnx_to_krnl/onnx_to_krnl_common.cpp
        conversion/onnx_to_krnl/onnx_to_krnl_common.hpp
        conversion/onnx_to_krnl/math/elementwise.cpp
        conversion/onnx_to_krnl/math/gemm.cpp
        conversion/onnx_to_krnl/math/matmul.cpp
        conversion/onnx_to_krnl/math/reduction.cpp
        conversion/onnx_to_krnl/math/softmax.cpp
        conversion/onnx_to_krnl/nn/conv.cpp
        conversion/onnx_to_krnl/nn/normalization.cpp
        conversion/onnx_to_krnl/nn/pooling.cpp
        conversion/onnx_to_krnl/tensor/identity.cpp
        conversion/onnx_to_krnl/tensor/reshape.cpp
        conversion/onnx_to_krnl/tensor/padconstantvaluepad.cpp
        conversion/onnx_to_krnl/tensor/transpose.cpp
        conversion/onnx_to_krnl/tensor/unsqueeze.cpp
        conversion/onnx_to_krnl/tensor/constant.cpp
        conversion/onnx_to_krnl/convert_onnx_to_krnl.cpp)
target_include_directories(onnx_mlir_lower_frontend
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_lower_frontend ${MLIRLibs})
add_dependencies(onnx_mlir_lower_frontend gen_krnl_ops)

add_subdirectory(transform)
add_subdirectory(tool)
add_subdirectory(builder)
add_subdirectory(runtime)

add_executable(onnx-mlir main.cpp)

target_link_libraries(onnx-mlir
        builder
        ${MLIRLibs}
        onnx_mlir_transform
        onnx_mlir_onnx_decompose
        onnx_mlir_shape_inference
        onnx_mlir_lower_frontend
        attribute_promotion)
whole_archive_link_mlir(onnx-mlir
        ${MLIRWholeArchiveLibs})

find_package(ZLIB REQUIRED)
target_link_libraries(onnx-mlir ${ZLIB_LIBRARIES})

target_include_directories(onnx-mlir PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS onnx-mlir DESTINATION bin)
