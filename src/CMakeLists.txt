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

set(ONNX_MLIR_LD_PRELOAD_onnx-mlir "" CACHE STRING "" FORCE)
whole_archive_link_mlir(onnx-mlir ${MLIRWholeArchiveLibs})
if(BUILD_SHARED_LIBS)
  message(STATUS "To run dynamically linked onnx-mlir, you must specify:")
  message(STATUS "LD_PRELOAD=${ONNX_MLIR_LD_PRELOAD_onnx-mlir}")
endif()

# Libraries specified on the target_link_libraries for the add_subdirectory
# targets get added to the end of the list here. This creates two problems:
# 1. It produces duplicated libraries being specified for the link command.
# 2. The libraries added at the end may depend on other libraries thus
#    cause linkage errors due to undefined symbols.
# So it's better not to use target_link_libraries for the add_subdirectory
# targets, or only use it for libraries that have no further dependencies
# (except system libraries such as libc).
target_link_libraries(onnx-mlir
        OMBuilder
        OMKrnlOps
        OMONNXOps
        OMShapeInference
        OMShapeInferenceOpInterface
        OMAttributePromotion
        OMPromotableConstOperandsOpInterface
        OMElideConstants
        OMElideKrnlGlobalConstants
        OMKrnlToAffine
        OMKrnlToLLVM
        OMONNXToKrnl
        OMONNXRewrite
        ${MLIRLibs}
        ${CMAKE_DL_LIBS})

if (INCLUDE_ONNX_ML)
  target_link_libraries(onnx-mlir OMMLONNXOps)
endif()


target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_BIN_ROOT})

install(TARGETS onnx-mlir DESTINATION bin)
