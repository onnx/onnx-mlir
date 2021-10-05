# SPDX-License-Identifier: Apache-2.0

include(GenerateExportHeader)

add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Builder)
add_subdirectory(Runtime)
add_subdirectory(Tool)

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ExternalUtil.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.hpp.cfg
  @ONLY
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/ExternalUtil.hpp
  INPUT ${CMAKE_CURRENT_BINARY_DIR}/ExternalUtil.hpp.cfg
  )

# CMAKE_CFG_INTDIR is . for single-config generators such as make, and
# it has a value (e.g. $(Configuration)) otherwise, so we can use it to
# determine whether we are dealing with a multi-config generator.
if (NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  set(FILE_GENERATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
else()
  set(FILE_GENERATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endif()

add_custom_target(ExternalUtil DEPENDS ${FILE_GENERATE_DIR}/ExternalUtil.hpp)

add_onnx_mlir_library(MainUtils
  MainUtils.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  ExternalUtil

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}

  LINK_LIBS PUBLIC
  ${OMLibs}
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  )

# MainUtils does not require cruntime or jniruntime to build, however, they are
# required for execution when using the EmitLib or EmitJNI options
add_dependencies(MainUtils cruntime)
if (ONNX_MLIR_ENABLE_JNI)
  add_dependencies(MainUtils jniruntime)
endif()

include_directories(${PROJECT_BINARY_DIR}/src)
add_onnx_mlir_library(om SHARED
  OMLibrary.cpp

  EXCLUDE_FROM_OM_LIBS
  
  LINK_LIBS PRIVATE
  MainUtils
  )

GENERATE_EXPORT_HEADER (om
    BASE_NAME onnx_mlir
    EXPORT_MACRO_NAME ONNX_MLIR_EXPORT
    EXPORT_FILE_NAME OMExport.hpp
    STATIC_DEFINE ONNX_MLIR_BUILT_AS_STATIC
)

add_onnx_mlir_executable(onnx-mlir
  main.cpp

  LINK_LIBS PRIVATE
  MainUtils
  )
