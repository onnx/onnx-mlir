add_executable(onnf-opt onnf_opt.cpp)

target_include_directories(onnf-opt PRIVATE ${ONNF_SRC_ROOT})
target_include_directories(onnf-opt PRIVATE ${ONNF_BIN_ROOT})

set(LIB_LIST
    MLIRStandardOps
    MLIRAffineOps
    MLIRLoopOps
    MLIRTransformUtils
    MLIREDSC
    MLIRTransforms)
whole_archive_link_mlir(onnf-opt ${LIB_LIST})

# TODO: need to investigate how to whole-archive link compiler pass to onnf-opt.
target_link_libraries(onnf-opt compiler)
