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

add_onnx_mlir_library(CompilerOptions
  CompilerOptions.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  Accelerator
  ExternalUtil
  AcceleratorsInc

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  MLIRIR
  Accelerator
  )

add_onnx_mlir_library(CompilerPasses
  CompilerPasses.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  MLIRIR

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  ${OMLibs}
  CompilerOptions
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  )

if (ONNX_MLIR_VENDOR)
  list(APPEND DEFINITIONS "ONNX_MLIR_VENDOR=\"${ONNX_MLIR_VENDOR}\"")
endif()

# ONNX_MLIR_PRODUCT_VERSION is specified/cached.
if (DEFINED ONNX_MLIR_PRODUCT_VERSION)
  string(REGEX MATCH "^([0-9]+)(\\.([0-9]+))?(\\.([0-9]+))?(\\-(.+))$"
	  VERSION_MATCH ${ONNX_MLIR_PRODUCT_VERSION})

  # For a version string 1.2.3-foo,
  # CMAKE_MATCH_1 = 1
  # CMAKE_MATCH_3 = 2
  # CMAKE_MATCH_5 = 3
  # CMAKE_MATCH_7 = foo
  set(PRODUCT_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(PRODUCT_VERSION_MINOR ${CMAKE_MATCH_3})
  set(PRODUCT_VERSION_PATCH ${CMAKE_MATCH_5})
  set(PRODUCT_ID ${CMAKE_MATCH_7})

  message(DEBUG "PRODUCT_VERSION_MAJOR = ${PRODUCT_VERSION_MAJOR}")
  message(DEBUG "PRODUCT_VERSION_MINOR = ${PRODUCT_VERSION_MINOR}")
  message(DEBUG "PRODUCT_VERSION_PATCH = ${PRODUCT_VERSION_PATCH}")
  message(DEBUG "PRODUCT_ID = ${PRODUCT_ID}")

  # Major and product ID are mandatory, minor and patch are optional.
  if (("${PRODUCT_VERSION_MAJOR}" STREQUAL "") OR ("${PRODUCT_ID}" STREQUAL ""))
    message(STATUS "Product version          : Invalid version ${ONNX_MLIR_PRODUCT_VERSION} ignored")
    unset(ONNX_MLIR_PRODUCT_VERSION CACHE)
  endif()
endif()

# When not specified or ignored, major, minor and patch are 0 and id is "NOT_SPECIFIED".
if (NOT DEFINED ONNX_MLIR_PRODUCT_VERSION OR "${PRODUCT_VERSION_MAJOR}" STREQUAL "")
  set(PRODUCT_VERSION_MAJOR 0)
endif()
if (NOT DEFINED ONNX_MLIR_PRODUCT_VERSION OR "${PRODUCT_VERSION_MINOR}" STREQUAL "")
  set(PRODUCT_VERSION_MINOR 0)
endif()
if (NOT DEFINED ONNX_MLIR_PRODUCT_VERSION OR "${PRODUCT_VERSION_PATCH}" STREQUAL "")
  set(PRODUCT_VERSION_PATCH 0)
endif()
if (NOT DEFINED ONNX_MLIR_PRODUCT_VERSION OR "${PRODUCT_ID}" STREQUAL "")
  set(PRODUCT_ID "NOT_SPECIFIED")
endif()
message(STATUS "Product version          : ${PRODUCT_VERSION_MAJOR}.${PRODUCT_VERSION_MINOR}.${PRODUCT_VERSION_PATCH}-${PRODUCT_ID}")

list(APPEND DEFINITIONS "PRODUCT_VERSION_MAJOR=${PRODUCT_VERSION_MAJOR}")
list(APPEND DEFINITIONS "PRODUCT_VERSION_MINOR=${PRODUCT_VERSION_MINOR}")
list(APPEND DEFINITIONS "PRODUCT_VERSION_PATCH=${PRODUCT_VERSION_PATCH}")
list(APPEND DEFINITIONS "PRODUCT_ID=\"${PRODUCT_ID}\"")
set_property(SOURCE CompilerUtils.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${DEFINITIONS})

add_onnx_mlir_library(CompilerUtils
  CompilerUtils.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  ExternalUtil
  MLIRIR
  OMVersion

  INCLUDE_DIRS PRIVATE
  ${FILE_GENERATE_DIR}
  
  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  ${OMLibs}
  CompilerPasses
  Accelerator
  InitAccelerators

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

if (NOT BUILD_SHARED_LIBS)
  target_compile_definitions(OnnxMlirCompiler PUBLIC ONNX_MLIR_BUILT_AS_STATIC)
endif()
