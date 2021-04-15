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
  OMKrnlToLLVM
  OMONNXToKrnl
  OMONNXRewrite
  OMShapeInference
  OMShapeInferenceOpInterface
  OMResultTypeInferenceOpInterface
  OMHasOnnxSubgraphOpInterface
  OMSpecializedKernelOpInterface
  OMElideConstants
  OMElideKrnlGlobalConstants
  OMEnableMemoryPool
  OMBundleMemoryPools
  OMOptimizeMemoryPools
  OMDisconnectKrnlDimFromAlloc
  OMLowerKrnlShape
  OMSimplifyKrnl
  )
set(OMLibs ${OMLibs} PARENT_SCOPE)

add_subdirectory(Tool)

add_library(MainUtils
  MainUtils.hpp
  MainUtils.cpp
  )

target_link_libraries(MainUtils
  PUBLIC
  ${OMLibs}
  ${MLIRLibs}
  ${CMAKE_DL_LIBS}
  onnx
  )

target_include_directories(MainUtils PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(MainUtils PRIVATE ${ONNX_MLIR_BIN_ROOT})
target_include_directories(MainUtils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MainUtils PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})

add_executable(onnx-mlir
        main.cpp)
target_link_libraries(onnx-mlir MainUtils)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ExternalUtil.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.hpp.cfg
  @ONLY
  )

if (NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/ExternalUtil.hpp
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.hpp.cfg
    )
else()
  file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.hpp
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.hpp.cfg
    )
endif()

add_custom_target(ExternalUtil
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/ExternalUtil.hpp
  )
add_dependencies(MainUtils ExternalUtil)

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

target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(onnx-mlir PRIVATE ${ONNX_MLIR_BIN_ROOT})

install(TARGETS onnx-mlir DESTINATION bin)
install(TARGETS MainUtils DESTINATION lib)
