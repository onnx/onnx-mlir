add_library(
        compiler
        dialect/krnl/krnl_ops.cpp
        dialect/krnl/krnl_ops.hpp
        dialect/krnl/krnl_types.cpp
        dialect/krnl/krnl_types.hpp
        dialect/onnx/onnx_ops.cpp
        dialect/onnx/onnx_ops.hpp
        dialect/krnl/parser_helper.cpp
        dialect/krnl/parser_helper.hpp
        pass/shape_inference_pass.cpp
        pass/shape_inference_interface.hpp
        pass/passes.hpp
        dialect/onnx/onnxop.inc
        pass/onnx_combine.cpp)

# Include root src directory.
target_include_directories(compiler PRIVATE ${ONNF_SRC_ROOT})

# Include third-party libraries.
target_include_directories(compiler PRIVATE ${ONNF_SRC_ROOT}/third_party/isl/include)
target_include_directories(compiler PRIVATE ${ONNF_BIN_ROOT}/third_party/isl/include)
target_include_directories(compiler PRIVATE ${ONNF_SRC_ROOT}/third_party/Linq)
target_include_directories(compiler PRIVATE ${ONNF_SRC_ROOT}/third_party/inja/src)
target_include_directories(compiler PRIVATE ${ONNF_SRC_ROOT}/third_party/fmt/include)

# Include tablegen generated header files.
target_include_directories(compiler PRIVATE ${ONNF_BIN_ROOT})

find_package(Boost 1.54.0
        COMPONENTS
        graph
        program_options
        log_setup
        log
        system
        filesystem
        REQUIRED)

target_link_libraries(compiler
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS}
        ${MLIRLIBS}
        curses)

add_subdirectory(tool)

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
add_public_tablegen_target(gen_onnx)
add_dependencies(compiler gen_onnx)

set(LLVM_TARGET_DEFINITIONS dialect/krnl/krnl_ops.td)
onnf_tablegen(krnl.hpp.inc -gen-op-decls)
onnf_tablegen(krnl.cpp.inc -gen-op-defs)
add_public_tablegen_target(gen_krnl_ops)
add_dependencies(compiler gen_krnl_ops)
add_dependencies(onnf-opt gen_krnl_ops)