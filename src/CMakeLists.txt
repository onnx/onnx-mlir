# SPDX-License-Identifier: Apache-2.0

add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Builder)
add_subdirectory(Runtime)

# All ONNX-MLIR libraries.
set(OMLibs
        OMBuilder
        # Note that circular dependencies exist between
        # ONNXOps and KrnlOps, so we include them twice
        # for full symbol resolution.
        OMONNXOps
        OMKrnlOps
        OMONNXOps
        OMKrnlOps
        OMSupport
        OMKrnlToAffine
        OMKrnlToStandard
        OMKrnlToLLVM
        OMONNXToKrnl
        OMONNXToLinalg
        OMONNXRewrite
        OMShapeInference
        OMShapeInferenceOpInterface
        OMAttributePromotion
        OMPromotableConstOperandsOpInterface
        OMResultTypeInferenceOpInterface
        OMHasOnnxSubgraphOpInterface
        OMElideConstants
        OMElideKrnlGlobalConstants
        OMPackKrnlGlobalConstants
        OMEnableMemoryPool
        OMBundleMemoryPools
        OMOptimizeMemoryPools
        OMDisconnectKrnlDimFromAlloc
        OMLowerKrnlShape
        OMSimplifyKrnl)
set(OMLibs ${OMLibs} PARENT_SCOPE)

add_subdirectory(Tool)

add_library(MainUtils
        MainUtils.hpp
        MainUtils.cpp)
target_link_libraries(MainUtils
        ${OMLibs}
        ${MLIRLibs}
        ${CMAKE_DL_LIBS}
        onnx)

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

# Libraries specified on the target_link_libraries for the add_subdirectory
# targets get added to the end of the list here. This creates two problems:
# 1. It produces duplicated libraries being specified for the link command.
# 2. The libraries added at the end may depend on other libraries thus
#    cause linkage errors due to undefined symbols.
# So it's better not to use target_link_libraries for the add_subdirectory
# targets, or only use it for libraries that have no further dependencies
# (except system libraries such as libc).
add_dependencies(onnx-mlir OMKrnlOpsInc OMONNXOpsInc)

add_dependencies(onnx-mlir cruntime)
add_dependencies(onnx-mlir EmbeddedDataLoader)

target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_BIN_ROOT})

install(TARGETS onnx-mlir DESTINATION bin)
install(TARGETS MainUtils DESTINATION lib)
