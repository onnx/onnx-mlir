# SPDX-License-Identifier: Apache-2.0

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

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

add_onnx_mlir_library(CompilerUtils
  CompilerUtils.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  ExternalUtil
  MLIRIR

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}
  
  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  ${OMLibs}
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation

  # Link LLVM libraries necessary to query which target architectures are configured.
  LINK_COMPONENTS PRIVATE
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  MC
  )

# CompilerUtils does not require cruntime or jniruntime to build, however, they are
# required for execution when using the EmitLib or EmitJNI options
add_dependencies(CompilerUtils cruntime)
if (ONNX_MLIR_ENABLE_JNI)
  add_dependencies(CompilerUtils jniruntime)
endif()

add_onnx_mlir_library(OnnxMlirCompiler
  OnnxMlirCompiler.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PRIVATE
  CompilerUtils
  )
