#add_library(compiler
#        Interface/ShapeInferenceInterface.hpp
#        Pass/Passes.hpp)

add_subdirectory(Interface)
add_subdirectory(Dialect)

## Include root src directory.
#target_include_directories(compiler PRIVATE ${ONNX_MLIR_SRC_ROOT})
#
## Include tablegen generated header files.
#target_include_directories(compiler PRIVATE ${ONNX_MLIR_BIN_ROOT})
#
#target_link_libraries(compiler
#        ${CMAKE_THREAD_LIBS_INIT}
#        ${CMAKE_DL_LIBS}
#        ${MLIRLibs}
#        curses)
#
##add_dependencies(gen_onnx gen_shape_inference)
#add_dependencies(compiler OMONNXOpsIncGen)
#
## TODO: onnx_mlir_gen_promotable_const_operands_op_interface is really a
## dependency of the onnx dialect library, which is currently part of `compiler`.
#add_dependencies(compiler onnx_mlir_gen_promotable_const_operands_op_interface)

add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Tool)
add_subdirectory(Builder)
add_subdirectory(Runtime)

add_executable(onnx-mlir main.cpp)

target_link_libraries(onnx-mlir
        OMBuilder
        ${MLIRLibs}
        OMKrnlOps
        OMONNXOps
        OMKrnlToAffine
        OMKrnlToLLVM
        OMShapeInference
        OMONNXToKrnl
        OMONNXRewrite
        onnx_mlir_attribute_promotion
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS})
whole_archive_link_mlir(onnx-mlir
        ${MLIRWholeArchiveLibs})

find_package(ZLIB REQUIRED)
target_link_libraries(onnx-mlir ${ZLIB_LIBRARIES})

target_include_directories(onnx-mlir PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS onnx-mlir DESTINATION bin)
