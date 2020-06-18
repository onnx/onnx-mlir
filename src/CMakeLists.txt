add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Tool)
add_subdirectory(Builder)
add_subdirectory(Runtime)

add_library(MainUtils
        MainUtils.hpp
        MainUtils.cpp)
target_link_libraries(MainUtils onnx)

target_include_directories(MainUtils PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(MainUtils PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(MainUtils PRIVATE ${ONNX_MLIR_BIN_ROOT})

add_executable(onnx-mlir
        main.cpp)
target_link_libraries(onnx-mlir MainUtils)

# Locate llc, which is needed for translating LLVM bitcode
# to object file.
if(NOT EXISTS "${LLVM_PROJ_BUILD}/bin/llc")
  message(ERROR "Cannot find llc.")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ExternalUtil.hpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.hpp)

set(ONNX_MLIR_LD_PRELOAD_onnx-mlir "" CACHE STRING "" FORCE)
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
target_link_libraries(MainUtils
        OMBuilder
        OMKrnlOps
        OMONNXOps
        OMShapeInference
        OMShapeInferenceOpInterface
        OMAttributePromotion
        OMPromotableConstOperandsOpInterface
        OMResultTypeInferenceOpInterface
        OMElideConstants
        OMElideKrnlGlobalConstants
        OMPackKrnlGlobalConstants
        OMEnableMemoryPool
        OMKrnlToAffine
        OMKrnlToLLVM
        OMONNXToKrnl
        OMONNXRewrite
        ${MLIRLibs}
        ${CMAKE_DL_LIBS})
add_dependencies(onnx-mlir OMKrnlOpsInc OMONNXOpsInc)

if (INCLUDE_ONNX_ML)
  target_link_libraries(MainUtils OMMLONNXOps)
  add_dependencies(MainUtils OMMLONNXOpsInc)
endif()

add_dependencies(onnx-mlir cruntime)
add_dependencies(onnx-mlir EmbeddedDataLoader)

target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_BIN_ROOT})

install(TARGETS onnx-mlir DESTINATION bin)
