add_library(builder
        sgir.cpp
        )

#Flags to link with LLVM/MLIR libraries
if(DEFINED ENV{LLVM_PROJECT_ROOT})
  set(LLVM_PROJECT_ROOT $ENV{LLVM_PROJECT_ROOT})
  if (EXISTS ${LLVM_PROJECT_ROOT})
    message(STATUS "LLVM_PROJECT_ROOT " ${LLVM_PROJECT_ROOT})
  else()
    message(FATAL_ERROR "The path specified by LLVM_PROJECT_ROOT does not exist: " ${LLVM_PROJECT_ROOT})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_PROJECT_ROOT not set")
endif()

if(DEFINED ENV{LLVM_PROJECT_LIB})
  set(LLVM_PROJECT_LIB $ENV{LLVM_PROJECT_LIB})
else()
  set(LLVM_PROJECT_LIB $ENV{LLVM_PROJECT_ROOT}/build/lib)
endif()
if (EXISTS ${LLVM_PROJECT_LIB})
  message(STATUS "LLVM_PROJECT_LIB " ${LLVM_PROJECT_LIB})
else()
  message(FATAL_ERROR "The path specified by LLVM_PROJECT_LIB does not exist: " ${LLVM_PROJECT_LIB})
endif()

#include path
include_directories(${LLVM_PROJECT_ROOT}/llvm/projects/mlir/include)
include_directories(${LLVM_PROJECT_ROOT}/llvm/include)
include_directories(${LLVM_PROJECT_ROOT}/build/include)
include_directories(${LLVM_PROJECT_ROOT}/build/projects/mlir/include)

find_library(MLIRLIBANALYSIS
  NAMES MLIRAnalysis
  PATHS ${LLVM_PROJECT_LIB}
  NO_DEFAULT_PATH)

find_library(MLIRLIBIR
  NAMES MLIRIR
  PATHS ${LLVM_PROJECT_LIB}
  NO_DEFAULT_PATH)

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

#libraries are set according to toy/Ch2
set(MLIRLIBS ${MLIRLIBANALYSIS} ${MLIRLIBIR} ${MLIRLIBPARSER} ${MLIRLIBTRANSFORMS}
  ${MLIRLIBANALYSIS} ${MLIRLIBVECTOROPS} ${MLIRLIBIR} ${MLIRLIBSUPPORT} ${MLIRLIBSTANDARDOPS}
  ${LLVMLIBSUPPORT})


target_link_libraries(builder compiler op onnx ${MLIRLIBS} curses)
target_include_directories(builder
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}/third_party/onnx
                           ${CMAKE_SOURCE_DIR})
