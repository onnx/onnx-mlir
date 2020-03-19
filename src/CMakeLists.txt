add_subdirectory(Interface)
add_subdirectory(Dialect)
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
        OMAttributePromotion
        ${CMAKE_THREAD_LIBS_INIT}
        ${CMAKE_DL_LIBS})
whole_archive_link_mlir(onnx-mlir
        ${MLIRWholeArchiveLibs})

find_package(ZLIB REQUIRED)
target_link_libraries(onnx-mlir ${ZLIB_LIBRARIES})

target_include_directories(onnx-mlir PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS onnx-mlir DESTINATION bin)
