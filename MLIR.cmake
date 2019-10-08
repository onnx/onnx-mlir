# Flags to link with LLVM/MLIR libraries
if(DEFINED ENV{LLVM_PROJECT_ROOT})
  set(LLVM_PROJECT_ROOT $ENV{LLVM_PROJECT_ROOT})
  if(EXISTS ${LLVM_PROJECT_ROOT})
    message(STATUS "LLVM_PROJECT_ROOT " ${LLVM_PROJECT_ROOT})
  else()
    message(
      FATAL_ERROR "The path specified by LLVM_PROJECT_ROOT does not exist: "
                  ${LLVM_PROJECT_ROOT})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_PROJECT_ROOT not set")
endif()

if(DEFINED ENV{LLVM_PROJECT_LIB})
  set(LLVM_PROJECT_LIB $ENV{LLVM_PROJECT_LIB})
else()
  set(LLVM_PROJECT_LIB $ENV{LLVM_PROJECT_ROOT}/build/lib)
endif()
if(EXISTS ${LLVM_PROJECT_LIB})
  message(STATUS "LLVM_PROJECT_LIB " ${LLVM_PROJECT_LIB})
else()
  message(FATAL_ERROR "The path specified by LLVM_PROJECT_LIB does not exist: "
                      ${LLVM_PROJECT_LIB})
endif()

# include path
set(LLVM_SRC_INCLUDE_PATH ${LLVM_PROJECT_ROOT}/llvm/include)
set(LLVM_BIN_INCLUDE_PATH ${LLVM_PROJECT_ROOT}/build/include)
set(MLIR_SRC_INCLUDE_PATH ${LLVM_PROJECT_ROOT}/llvm/projects/mlir/include)
set(MLIR_BIN_INCLUDE_PATH ${LLVM_PROJECT_ROOT}/build/projects/mlir/include)

set(MLIR_INCLUDE_PATHS
    ${LLVM_SRC_INCLUDE_PATH};${LLVM_BIN_INCLUDE_PATH};${MLIR_SRC_INCLUDE_PATH};${MLIR_BIN_INCLUDE_PATH})
include_directories(${MLIR_INCLUDE_PATHS})

find_library(MLIRLIBANALYSIS
             NAMES MLIRAnalysis
             PATHS ${LLVM_PROJECT_LIB}
             NO_DEFAULT_PATH)

find_library(MLIRLIBIR NAMES MLIRIR PATHS ${LLVM_PROJECT_LIB} NO_DEFAULT_PATH)

find_library(MLIRLIBPARSER
             NAMES MLIRParser
             PATHS ${LLVM_PROJECT_LIB}
             NO_DEFAULT_PATH)

find_library(MLIRLIBTRANSFORMS
             NAMES MLIRTransforms
             PATHS ${LLVM_PROJECT_LIB}
             NO_DEFAULT_PATH)

find_library(MLIRLIBVECTOROPS
             NAMES MLIRVectorOps
             PATHS ${LLVM_PROJECT_LIB}
             NO_DEFAULT_PATH)

find_library(MLIRLIBSUPPORT
             NAMES MLIRSupport
             PATHS ${LLVM_PROJECT_LIB}
             NO_DEFAULT_PATH)

find_library(MLIRLIBSTANDARDOPS
             NAMES MLIRStandardOps
             PATHS ${LLVM_PROJECT_LIB}
             NO_DEFAULT_PATH)

find_library(LLVMLIBSUPPORT
             NAMES LLVMSupport
             PATHS ${LLVM_PROJECT_LIB}
             NO_DEFAULT_PATH)

# libraries are set according to toy/Ch2
set(MLIRLIBS
    ${MLIRLIBANALYSIS}
    ${MLIRLIBIR}
    ${MLIRLIBPARSER}
    ${MLIRLIBTRANSFORMS}
    ${MLIRLIBANALYSIS}
    ${MLIRLIBVECTOROPS}
    ${MLIRLIBIR}
    ${MLIRLIBSUPPORT}
    ${MLIRLIBSTANDARDOPS}
    ${LLVMLIBSUPPORT})

# Set up TableGen environment.
include(${LLVM_PROJECT_ROOT}/build/lib/cmake/llvm/TableGen.cmake)

function(onnf_tablegen ofn)
  tablegen(MLIR
           ${ARGV}
           "-I${MLIR_SRC_INCLUDE_PATH}"
           "-I${MLIR_BIN_INCLUDE_PATH}")
  set(TABLEGEN_OUTPUT
      ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

# Import the pre-built mlir TableGen as an imported exetuable. It is required by
# the LLVM TableGen command to have the TableGen target so that changes to the
# table gen utility itself can be detected and cause re-compilation of .td file.
add_executable(mlir-tblgen IMPORTED)
set_property(TARGET mlir-tblgen
             PROPERTY IMPORTED_LOCATION
                      ${LLVM_PROJECT_ROOT}/build/bin/mlir-tblgen)
set(MLIR_TABLEGEN_EXE mlir-tblgen)
