# SPDX-License-Identifier: Apache-2.0

# Path to LLVM build folder.
if(DEFINED LLVM_PROJ_BUILD)
  # Just use it...
elseif(DEFINED ENV{LLVM_PROJ_BUILD})
  set(LLVM_PROJ_BUILD $ENV{LLVM_PROJ_BUILD})
else()
  message(FATAL_ERROR "LLVM_PROJ_BUILD is not configured.  Please set the env variable "
  "LLVM_PROJ_BUILD or the corresponding cmake configuration option to reference an LLVM build.")
endif()
if(EXISTS ${LLVM_PROJ_BUILD})
  message(STATUS "LLVM_PROJ_BUILD         : " ${LLVM_PROJ_BUILD})
else()
  message(FATAL_ERROR "The path specified by LLVM_PROJ_BUILD does not exist: "
        ${LLVM_PROJ_BUILD})
endif()

set(LLVM_CMAKE_DIR
      "${LLVM_PROJ_BUILD}/lib/cmake/llvm"
      CACHE PATH "Path to LLVM cmake modules")

# Variables used to find cmake export information
set(LLVM_DIR ${LLVM_CMAKE_DIR})
set(MLIR_DIR ${LLVM_CMAKE_DIR}/../mlir)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)



# LLVM project lib folder
if (ENV{LLVM_PROJECT_LIB})
  set(LLVM_PROJECT_LIB $ENV{LLVM_PROJECT_LIB})
else()
  if(MSVC)
    if (CMAKE_BUILD_TYPE)
      set(LLVM_PROJECT_LIB ${LLVM_PROJ_BUILD}/${CMAKE_BUILD_TYPE}/lib)
    else()
      set(LLVM_PROJECT_LIB ${LLVM_PROJ_BUILD}/Release/lib)
    endif()  
  else()
    set(LLVM_PROJECT_LIB ${LLVM_PROJ_BUILD}/lib)
  endif()
endif()
message(STATUS "LLVM_PROJECT_LIB        : " ${LLVM_PROJECT_LIB})

# LLVM project bin folder
if (ENV{LLVM_PROJ_BIN})
  set(LLVM_PROJ_BIN $ENV{LLVM_PROJ_BIN})
else()
  set(LLVM_PROJ_BIN ${LLVM_TOOLS_BINARY_DIR})
endif()
message(STATUS "LLVM_PROJ_BIN           : " ${LLVM_PROJ_BIN})

# FIXME: Remove after migration
set(MLIR_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

# ONNX-MLIR tools folder
set(ONNX_MLIR_TOOLS_DIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR})
message(STATUS "ONNX_MLIR_TOOLS_DIR     : " ${ONNX_MLIR_TOOLS_DIR})
# Set output library path
set(ONNX_MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})

set(ONNX_MLIR_LIT_TEST_SRC_DIR ${ONNX_MLIR_SRC_ROOT}/test/mlir)
set(ONNX_MLIR_LIT_TEST_BUILD_DIR ${CMAKE_BINARY_DIR}/test/mlir)

include_directories(${LLVM_INCLUDE_DIRS};${MLIR_INCLUDE_DIRS})

# If we're pointing at an LLVM build, then assume we want to install into the same
# install location
if(EXISTS ${LLVM_PROJ_BUILD}/CMakeCache.txt)
  file(STRINGS ${LLVM_PROJ_BUILD}/CMakeCache.txt prefix REGEX CMAKE_INSTALL_PREFIX)
  string(REGEX REPLACE "CMAKE_INSTALL_PREFIX:PATH=" "" prefix ${prefix})
  string(REGEX REPLACE "//.*" "" prefix ${prefix})
  set(CMAKE_INSTALL_PREFIX ${prefix} CACHE PATH "" FORCE)
  message(STATUS "CMAKE_INSTALL_PREFIX    : " ${CMAKE_INSTALL_PREFIX})
endif()

# Force BUILD_SHARED_LIBS to be the same as LLVM build
set(shared ${LLVM_ENABLE_SHARED_LIBS})
set(BUILD_SHARED_LIBS ${shared} CACHE BOOL "" FORCE)
message(STATUS "BUILD_SHARED_LIBS       : " ${BUILD_SHARED_LIBS})

# Threading libraries required due to parallel pass execution.
# FIXME Should be added to LLVMConfig.cmake
find_package(Threads REQUIRED)
set(MLIR_SYSTEM_LIBS ${CMAKE_THREAD_LIBS_INIT})

# libcurses and libz required by libLLVMSupport on non-windows platforms
if(NOT MSVC)
  # FIXME Should be added to LLVMConfig.cmake
  find_package(Curses REQUIRED)
  set(MLIR_SYSTEM_LIBS ${MLIR_SYSTEM_LIBS} ${ZLIB_LIBRARIES} ${CURSES_LIBRARIES})
endif()

# FIXME: Remove after migration
#set(MLIRLibs ${MLIR_ALL_LIBS})
get_property(MLIRLibs GLOBAL PROPERTY MLIR_ALL_LIBS)

function(onnx_mlir_tablegen ofn)
  include_directories(${ONNX_MLIR_SRC_ROOT})
  tablegen(MLIR
          ${ARGV})
  set(TABLEGEN_OUTPUT
          ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
          PARENT_SCOPE)
endfunction()

# Import the pre-built mlir TableGen as an imported exetuable. It is required by
# the LLVM TableGen command to have the TableGen target so that changes to the
# table gen utility itself can be detected and cause re-compilation of .td file.
if (NOT TARGET mlir-tblgen)
  add_executable(mlir-tblgen IMPORTED)
  # Specify extension for incremental Windows builds.
  if(MSVC)
    set_property(TARGET mlir-tblgen
          PROPERTY IMPORTED_LOCATION ${LLVM_PROJ_BIN}/mlir-tblgen.exe)
  else()
    set_property(TARGET mlir-tblgen
          PROPERTY IMPORTED_LOCATION ${LLVM_PROJ_BIN}/mlir-tblgen)
  endif()
endif()

# Add a dialect used by ONNX MLIR and copy the generated operation
# documentation to the desired places.
# c.f. https://github.com/llvm/llvm-project/blob/e298e216501abf38b44e690d2b28fc788ffc96cf/mlir/CMakeLists.txt#L11
function(add_onnx_mlir_dialect_doc dialect dialect_tablegen_file)
  # Generate Dialect Documentation
  set(LLVM_TARGET_DEFINITIONS ${dialect_tablegen_file})
  onnx_mlir_tablegen(${dialect}.md -gen-op-doc)
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
