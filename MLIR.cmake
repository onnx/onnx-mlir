# Path to LLVM source folder.
if(DEFINED ENV{LLVM_SRC})
  set(LLVM_SRC $ENV{LLVM_SRC})
  if(EXISTS ${LLVM_SRC})
    message(STATUS "LLVM_SRC " ${LLVM_SRC})
  else()
    message(FATAL_ERROR "The path specified by LLVM_SRC does not exist: "
                        ${LLVM_SRC})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_SRC not set")
endif()

# Path to LLVM build folder
if(DEFINED ENV{LLVM_BUILD})
  set(LLVM_BUILD $ENV{LLVM_BUILD})
  if(EXISTS ${LLVM_BUILD})
    message(STATUS "LLVM_BUILD " ${LLVM_BUILD})
  else()
    message(FATAL_ERROR "The path specified by LLVM_BUILD does not exist: "
                        ${LLVM_BUILD})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_BUILD not set")
endif()

# LLVM project lib folder
set(LLVM_PROJECT_LIB ${LLVM_BUILD}/lib)

# Include paths for MLIR
set(LLVM_SRC_INCLUDE_PATH ${LLVM_SRC}/include)
set(LLVM_BIN_INCLUDE_PATH ${LLVM_BUILD}/include)
set(MLIR_SRC_INCLUDE_PATH ${LLVM_SRC}/projects/mlir/include)
set(MLIR_BIN_INCLUDE_PATH ${LLVM_BUILD}/projects/mlir/include)
set(MLIR_TOOLS_DIR ${LLVM_BUILD}/bin)

set(ONNF_TOOLS_DIR ${ONNF_BIN_ROOT}/src/compiler/tool/onnf_opt)
set(ONNF_LIT_TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/test/mlir)
set(ONNF_LIT_TEST_BUILD_DIR ${CMAKE_BINARY_DIR}/test/mlir)

set(
  MLIR_INCLUDE_PATHS
  ${LLVM_SRC_INCLUDE_PATH};${LLVM_BIN_INCLUDE_PATH};${MLIR_SRC_INCLUDE_PATH};${MLIR_BIN_INCLUDE_PATH}
  )
include_directories(${MLIR_INCLUDE_PATHS})

# Threading libraries required due to parallel pass execution.
find_package(Threads REQUIRED)

function(find_mlir_lib lib)
  find_library(${lib}
               NAMES ${lib}
               PATHS ${LLVM_PROJECT_LIB}
               NO_DEFAULT_PATH)
endfunction(find_mlir_lib)

find_mlir_lib(MLIRAffineOps)
find_mlir_lib(MLIRAffineToStandard)
find_mlir_lib(MLIRAnalysis)
find_mlir_lib(MLIRExecutionEngine)
find_mlir_lib(MLIRIR)
find_mlir_lib(MLIRLLVMIR)
find_mlir_lib(MLIRLoopToStandard)
find_mlir_lib(MLIRParser)
find_mlir_lib(MLIRPass)
find_mlir_lib(MLIRStandardOps)
find_mlir_lib(MLIRStandardToLLVM)
find_mlir_lib(MLIRTargetLLVMIR)
find_mlir_lib(MLIRTransforms)
find_mlir_lib(MLIRTransforms)
find_mlir_lib(MLIRTransformUtils)
find_mlir_lib(MLIRSupport)
find_mlir_lib(MLIROptMain)

find_mlir_lib(LLVMCore)
find_mlir_lib(LLVMSupport)
find_mlir_lib(LLVMAsmParser)
find_mlir_lib(LLVMBinaryFormat)
find_mlir_lib(LLVMRemarks)
find_mlir_lib(LLVMIRReader)
find_mlir_lib(LLVMTransformUtils)
find_mlir_lib(LLVMBitstreamReader)

set(MLIRLibsOnce
        MLIRAffineOps
        MLIRAffineToStandard
        MLIRAnalysis
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMIR
        MLIRLoopToStandard
        MLIRParser
        MLIRPass
        MLIRStandardOps
        MLIRStandardToLLVM
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRAffineOps
        MLIRAffineToStandard
        MLIRAnalysis
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMIR
        MLIRLoopToStandard
        MLIRParser
        MLIRPass
        MLIRStandardOps
        MLIRStandardToLLVM
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRTransformUtils
        MLIRLoopOps
        MLIRSupport
        MLIROptMain
        LLVMCore
        LLVMSupport
        LLVMAsmParser
        LLVMIRReader
        LLVMTransformUtils
        LLVMBinaryFormat
        LLVMRemarks
        LLVMBitstreamReader)

set(MLIRLibs
        ${MLIRLibsOnce}
        ${MLIRLibsOnce}
        Threads::Threads)

set(MLIRWholeArchiveLibs
        MLIRAffineToStandard
        MLIRAffineOps
        MLIRLLVMIR
        MLIRStandardOps
        MLIRStandardToLLVM
        MLIRLoopToStandard)

function(whole_archive_link target lib_dir)
  get_property(link_flags TARGET ${target} PROPERTY LINK_FLAGS)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(link_flags "${link_flags} -L${lib_dir} ")
    foreach(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags}
                    "-Wl,-force_load ${lib_dir}/lib${LIB}.a ")
    endforeach(LIB)
  elseif(MSVC)
    foreach(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "/WHOLEARCHIVE:${LIB} ")
    endforeach(LIB)
  else()
    set(link_flags "${link_flags} -L${lib_dir} -Wl,--whole-archive,")
    foreach(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-l${LIB},")
    endforeach(LIB)
    string(CONCAT link_flags ${link_flags} "--no-whole-archive")
  endif()
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

function(whole_archive_link_mlir target)
  whole_archive_link(${target} ${LLVM_BUILD}/lib ${ARGN})
endfunction(whole_archive_link_mlir)

function(whole_archive_link_onnf target)
  foreach(LIB ${ARGN})
    add_dependencies(${target} ${LIB})
  endforeach(LIB)
  whole_archive_link(${target} ${CMAKE_BINARY_DIR}/lib ${ARGN})
endfunction(whole_archive_link_onnf)

set(LLVM_CMAKE_DIR
    "${LLVM_BUILD}/lib/cmake/llvm"
    CACHE PATH "Path to LLVM cmake modules")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)

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
             PROPERTY IMPORTED_LOCATION ${LLVM_BUILD}/bin/mlir-tblgen)
set(MLIR_TABLEGEN_EXE mlir-tblgen)
