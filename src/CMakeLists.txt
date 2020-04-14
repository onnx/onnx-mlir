add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Tool)
add_subdirectory(Builder)
add_subdirectory(Runtime)

add_executable(onnx-mlir
        MainUtils.hpp
        MainUtils.cpp
        main.cpp)
target_link_libraries(onnx-mlir
        ${MLIRLibs}
        OMBuilder
        OMKrnlOps
        OMONNXOps
        OMKrnlToAffine
        OMKrnlToLLVM
        OMShapeInference
        OMONNXToKrnl
        OMONNXRewrite
        OMAttributePromotion
        OMElideConstants
        OMElideKrnlGlobalConstants
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS})
whole_archive_link_mlir(onnx-mlir
        ${MLIRWholeArchiveLibs})

find_package(ZLIB REQUIRED)
target_link_libraries(onnx-mlir ${ZLIB_LIBRARIES})

target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_BIN_ROOT})

install(TARGETS onnx-mlir DESTINATION bin)
