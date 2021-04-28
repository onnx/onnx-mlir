# SPDX-License-Identifier: Apache-2.0

if (DEFINED ENV{MLIR_DIR})
  set(MLIR_DIR $ENV{MLIR_DIR} CACHE PATH "Path to directory containing MLIRConfig.cmake")
elseif (NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR "MLIR_DIR is not configured but it is required. "
          "Please set the env variable MLIR_DIR or the corresponding cmake configuration option.")
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# This is the list of all MLIR libs needed by *any* of the libs or execurables
# in the project. Making a list of all the libs here most closely matches the
# existing behavior, but in the future, each of these libs can be added as
# dependencies in the appropriate locations removing the need for a centralized
# list.
set(MLIRLibs
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIROptLib
  MLIRSCFToStandard
  MLIRShapeToStandard
  MLIRTargetLLVMIR
)

set(BUILD_SHARED_LIBS ${LLVM_ENABLE_SHARED_LIBS} CACHE BOOL "" FORCE)
message(STATUS "BUILD_SHARED_LIBS       : " ${BUILD_SHARED_LIBS})

# If CMAKE_INSTALL_PREFIX was not provided explicitly and we are not using an install of
# LLVM and a CMakeCache.txt exists,
# force CMAKE_INSTALL_PREFIX to be the same as the LLVM build.
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND NOT LLVM_INSTALL_PREFIX)
  if (EXISTS ${LLVM_BINARY_DIR}/CMakeCache.txt)
    file(STRINGS ${LLVM_BINARY_DIR}/CMakeCache.txt prefix REGEX CMAKE_INSTALL_PREFIX)
    string(REGEX REPLACE "CMAKE_INSTALL_PREFIX:PATH=" "" prefix ${prefix})
    string(REGEX REPLACE "//.*" "" prefix ${prefix})
    set(CMAKE_INSTALL_PREFIX ${prefix} CACHE PATH "" FORCE)
  endif()
endif()
message(STATUS "CMAKE_INSTALL_PREFIX    : " ${CMAKE_INSTALL_PREFIX})

# The tablegen functions below are modeled based on the corresponding functions
# in mlir: https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIR.cmake
function(add_onnx_mlir_dialect_doc dialect dialect_tablegen_file)
  # Generate Dialect Documentation
  set(LLVM_TARGET_DEFINITIONS ${dialect_tablegen_file})
  tablegen(MLIR ${dialect}.md -gen-op-doc "-I${ONNX_MLIR_SRC_ROOT}")
  set(GEN_DOC_FILE ${ONNX_MLIR_BIN_ROOT}/docs/Dialects/${dialect}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md)
  add_custom_target(${dialect}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(onnx-mlir-doc ${dialect}DocGen)
endfunction()
add_custom_target(onnx-mlir-doc)

function(add_onnx_mlir_dialect dialect)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.hpp.inc -gen-op-decls "-I${ONNX_MLIR_SRC_ROOT}")
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs "-I${ONNX_MLIR_SRC_ROOT}")
  add_public_tablegen_target(OM${dialect}IncGen)
endfunction()

function(add_onnx_mlir_rewriter rewriter)
  set(LLVM_TARGET_DEFINITIONS ${rewriter}.td)
  mlir_tablegen(ONNX${rewriter}.inc -gen-rewriters "-I${ONNX_MLIR_SRC_ROOT}")
  add_public_tablegen_target(OMONNX${rewriter}IncGen)
endfunction()

function(add_onnx_mlir_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.hpp.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(OM${interface}IncGen)
endfunction()
