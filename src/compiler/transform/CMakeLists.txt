add_library(onnf_transform
            lower_krnl.cpp
            lower_to_llvm.cpp)

target_include_directories(onnf_transform
                           PRIVATE ${ONNF_SRC_ROOT} ${ONNF_BIN_ROOT}
                                   ${ONNF_SRC_ROOT})
target_link_libraries(onnf_transform ${MLIRLibs})
add_dependencies(onnf_transform gen_krnl_ops)
