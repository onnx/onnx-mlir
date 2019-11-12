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
        pass/passes.hpp)

# Include root src directory.
target_include_directories(compiler PRIVATE ../..)
target_include_directories(compiler PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(compiler PRIVATE ${CMAKE_BINARY_DIR})

find_package(Boost 1.54.0
        COMPONENTS graph
        program_options
        log_setup
        log
        system
        filesystem
        REQUIRED)

# target_link_libraries(compiler isl inja ${Boost_LIBRARIES})
target_link_libraries(compiler
        ${Boost_LIBRARIES}
        ${MLIRLIBS} curses)

add_executable(onnf-opt
        tool/onnf_opt/onnf_opt.cpp)

target_link_libraries(onnf-opt ${Boost_LIBRARIES} ${MLIRLIBS} curses compiler)
target_include_directories(onnf-opt PRIVATE ../..)
target_include_directories(onnf-opt PRIVATE ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS pass/shape_inference_interface.td)
onnf_tablegen(shape_inference.hpp.inc -gen-op-interface-decls)
onnf_tablegen(shape_inference.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(gen_shape_inference)
add_dependencies(compiler gen_shape_inference)

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
