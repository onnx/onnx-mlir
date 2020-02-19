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
        dialect/onnx/onnxop.inc
        pass/onnx_combine.cpp
        pass/onnx_rewrite.cpp
        pass/passes.hpp)

# Include root src directory.
target_include_directories(compiler PRIVATE ${ONNF_SRC_ROOT})

# Include tablegen generated header files.
target_include_directories(compiler PRIVATE ${ONNF_BIN_ROOT})

target_link_libraries(compiler
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        ${MLIRLibs}
        curses)

set(LLVM_TARGET_DEFINITIONS pass/shape_inference_interface.td)
onnf_tablegen(shape_inference.hpp.inc -gen-op-interface-decls)
onnf_tablegen(shape_inference.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(gen_shape_inference)
add_dependencies(compiler gen_shape_inference)

set(LLVM_TARGET_DEFINITIONS pass/onnx_combine.td)
onnf_tablegen(onnx_combine.inc -gen-rewriters)
add_public_tablegen_target(gen_onnx_combine)
add_dependencies(compiler gen_onnx_combine)

set(LLVM_TARGET_DEFINITIONS dialect/onnx/onnx.td)
onnf_tablegen(onnx.hpp.inc -gen-op-decls "-I${CMAKE_SOURCE_DIR}/compiler/pass")
onnf_tablegen(onnx.cpp.inc -gen-op-defs "-I${CMAKE_SOURCE_DIR}/compiler/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/Dialects/onnx.md)
add_public_tablegen_target(gen_onnx)
add_dependencies(compiler gen_onnx)
add_onnf_dialect_doc(onnx dialect/onnx/onnx.td)

set(LLVM_TARGET_DEFINITIONS dialect/krnl/krnl_ops.td)
onnf_tablegen(krnl.hpp.inc -gen-op-decls)
onnf_tablegen(krnl.cpp.inc -gen-op-defs)
add_public_tablegen_target(gen_krnl_ops)
add_dependencies(compiler gen_krnl_ops)

add_library(onnf_shape_inference pass/shape_inference_pass.cpp)
target_include_directories(onnf_shape_inference
        PRIVATE ${ONNF_SRC_ROOT} ${ONNF_BIN_ROOT}
        ${ONNF_SRC_ROOT})
target_link_libraries(onnf_shape_inference ${MLIRLibs})
add_dependencies(onnf_shape_inference gen_krnl_ops)

add_library(onnf_lower_frontend conversion/onnx_to_krnl/convert_onnx_to_krnl.cpp)
target_include_directories(onnf_lower_frontend
        PRIVATE ${ONNF_SRC_ROOT} ${ONNF_BIN_ROOT}
        ${ONNF_SRC_ROOT})
target_link_libraries(onnf_lower_frontend ${MLIRLibs})
add_dependencies(onnf_lower_frontend gen_krnl_ops)

add_subdirectory(transform)
add_subdirectory(tool)
add_subdirectory(builder)
add_subdirectory(runtime)

add_executable(onnf main.cpp)

target_link_libraries(onnf builder ${MLIRLibs} onnf_transform onnf_shape_inference onnf_lower_frontend)
whole_archive_link_mlir(onnf ${MLIRWholeArchiveLibs})
find_package(ZLIB REQUIRED)
target_link_libraries(onnf ${ZLIB_LIBRARIES})

target_include_directories(onnf PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnf PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS onnf DESTINATION bin)
