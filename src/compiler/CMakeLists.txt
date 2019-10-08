add_library(
        compiler
        ir/knl/knl_ops.cpp
        ir/knl/knl_ops.hpp)

# Include root src directory.
target_include_directories(compiler PRIVATE ../..)
target_include_directories(compiler PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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
        )

set(LLVM_TARGET_DEFINITIONS ir/knl/knl.td)
onnf_tablegen(knl.hpp.inc -gen-op-decls)
onnf_tablegen(knl.cpp.inc -gen-op-defs)
add_public_tablegen_target(gen_kir)
add_dependencies(compiler gen_kir)
