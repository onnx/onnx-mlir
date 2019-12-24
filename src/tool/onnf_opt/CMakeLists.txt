add_executable(onnf-opt onnf_opt.cpp)
add_dependencies(onnf-opt gen_krnl_ops)

target_include_directories(onnf-opt PRIVATE ${ONNF_SRC_ROOT})
target_include_directories(onnf-opt PRIVATE ${ONNF_BIN_ROOT})

target_link_libraries(onnf-opt ${MLIRLibs} curses)
whole_archive_link_mlir(onnf-opt ${MLIRWholeArchiveLibs})
whole_archive_link_onnf(onnf-opt compiler onnf_transform onnf_lower_frontend onnf_shape_inference)