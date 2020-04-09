# Path to LLVM source folder.
if(DEFINED ENV{LLVM_PROJ_SRC})
  set(LLVM_PROJ_SRC $ENV{LLVM_PROJ_SRC})
  if(EXISTS ${LLVM_PROJ_SRC})
    message(STATUS "LLVM_PROJ_SRC " ${LLVM_PROJ_SRC})
  else()
    message(FATAL_ERROR "The path specified by LLVM_PROJ_SRC does not exist: "
            ${LLVM_PROJ_SRC})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_PROJ_SRC not set")
endif()

# Path to LLVM build folder
if(DEFINED ENV{LLVM_PROJ_BUILD})
  set(LLVM_PROJ_BUILD $ENV{LLVM_PROJ_BUILD})
  if(EXISTS ${LLVM_PROJ_BUILD})
    message(STATUS "LLVM_PROJ_BUILD " ${LLVM_PROJ_BUILD})
  else()
    message(FATAL_ERROR "The path specified by LLVM_PROJ_BUILD does not exist: "
            ${LLVM_PROJ_BUILD})
  endif()
else()
  message(FATAL_ERROR "env variable LLVM_PROJ_BUILD not set")
endif()

# LLVM project lib folder
set(LLVM_PROJECT_LIB ${LLVM_PROJ_BUILD}/lib)

# Include paths for MLIR
set(LLVM_SRC_INCLUDE_PATH ${LLVM_PROJ_SRC}/llvm/include)
set(LLVM_BIN_INCLUDE_PATH ${LLVM_PROJ_BUILD}/include)
set(MLIR_SRC_INCLUDE_PATH ${LLVM_PROJ_SRC}/mlir/include)
set(MLIR_BIN_INCLUDE_PATH ${LLVM_PROJ_BUILD}/tools/mlir/include)
set(MLIR_TOOLS_DIR ${LLVM_PROJ_BUILD}/bin)

set(ONNX_MLIR_TOOLS_DIR ${ONNX_MLIR_BIN_ROOT}/bin)
set(ONNX_MLIR_LIT_TEST_SRC_DIR ${ONNX_MLIR_SRC_ROOT}/test/mlir)
set(ONNX_MLIR_LIT_TEST_BUILD_DIR ${CMAKE_BINARY_DIR}/test/mlir)

set(
        MLIR_INCLUDE_PATHS
        ${LLVM_SRC_INCLUDE_PATH};${LLVM_BIN_INCLUDE_PATH};${MLIR_SRC_INCLUDE_PATH};${MLIR_BIN_INCLUDE_PATH}
)
include_directories(${MLIR_INCLUDE_PATHS})

# Force BUILD_SHARED_LIBS to be the same as LLVM build
file(STRINGS ${LLVM_PROJ_BUILD}/CMakeCache.txt shared REGEX BUILD_SHARED_LIBS)
string(REGEX REPLACE "BUILD_SHARED_LIBS:BOOL=" "" shared ${shared})
set(BUILD_SHARED_LIBS ${shared} CACHE BOOL "" FORCE)
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")

# Threading libraries required due to parallel pass execution.
find_package(Threads REQUIRED)
# libcurses and libz required by libLLVMSupport
find_package(Curses REQUIRED)
find_package(ZLIB REQUIRED)

function(find_mlir_lib lib)
  find_library(${lib}
          NAMES ${lib}
          PATHS ${LLVM_PROJECT_LIB}
          NO_DEFAULT_PATH)
  if(${${lib}} STREQUAL ${lib}-NOTFOUND)
    message(FATAL_ERROR "${lib} not found")
    #else()
    #set(${lib} "-l${lib}" PARENT_SCOPE)
  endif()
endfunction(find_mlir_lib)

find_mlir_lib(MLIRAffine)
find_mlir_lib(MLIRAffineToStandard)
find_mlir_lib(MLIRAnalysis)
find_mlir_lib(MLIRCallInterfaces)
find_mlir_lib(MLIRControlFlowInterfaces)
find_mlir_lib(MLIRDialect)
find_mlir_lib(MLIREDSC)
find_mlir_lib(MLIRExecutionEngine)
find_mlir_lib(MLIRIR)
find_mlir_lib(MLIRLLVMIR)
find_mlir_lib(MLIRLoopAnalysis)
find_mlir_lib(MLIRLoopToStandard)
find_mlir_lib(MLIRLoopOps)
find_mlir_lib(MLIRLoopLikeInterface)
find_mlir_lib(MLIRLLVMIRTransforms)
find_mlir_lib(MLIRMlirOptMain)
find_mlir_lib(MLIRParser)
find_mlir_lib(MLIRPass)
find_mlir_lib(MLIRStandardOps)
find_mlir_lib(MLIRStandardToLLVM)
find_mlir_lib(MLIRSideEffects)
find_mlir_lib(MLIRTargetLLVMIR)
find_mlir_lib(MLIRTransforms)
find_mlir_lib(MLIRTransformUtils)
find_mlir_lib(MLIRSupport)
find_mlir_lib(MLIROpenMP)
find_mlir_lib(MLIROptLib)
find_mlir_lib(MLIRTargetLLVMIRModuleTranslation)
find_mlir_lib(MLIRTargetLLVMIR)
find_mlir_lib(MLIRTransformUtils)
find_mlir_lib(MLIRTranslation)
find_mlir_lib(MLIRVector)

find_mlir_lib(LLVMCore)
find_mlir_lib(LLVMSupport)
find_mlir_lib(LLVMAsmParser)
find_mlir_lib(LLVMBinaryFormat)
find_mlir_lib(LLVMRemarks)
find_mlir_lib(LLVMIRReader)
find_mlir_lib(LLVMMLIRTableGen)
find_mlir_lib(LLVMTransformUtils)
find_mlir_lib(LLVMBitstreamReader)
find_mlir_lib(LLVMAnalysis)
find_mlir_lib(LLVMBitWriter)
find_mlir_lib(LLVMBitReader)
find_mlir_lib(LLVMMC)
find_mlir_lib(LLVMMCParser)
find_mlir_lib(LLVMObject)
find_mlir_lib(LLVMProfileData)
find_mlir_lib(LLVMDemangle)
find_mlir_lib(LLVMFrontendOpenMP)

set(MLIRLibs
        ${MLIRLLVMIR}
        ${MLIROptLib}
        ${MLIRParser}
        ${MLIRPass}
        ${MLIRTargetLLVMIR}
        ${MLIRTargetLLVMIRModuleTranslation}
        ${MLIRTransforms}
        ${MLIRTransformUtils}
        ${MLIRAffine}
        ${MLIRAffineToStandard}
        ${MLIRAnalysis}
        ${MLIRCallInterfaces}
        ${MLIRControlFlowInterfaces}
        ${MLIRDialect}
        ${MLIREDSC}
        ${MLIRExecutionEngine}
        ${MLIRIR}
        ${MLIRLLVMIRTransforms}        
        ${MLIRLoopToStandard}
        ${MLIRLoopOps}
        ${MLIRLoopAnalysis}
        ${MLIRLoopLikeInterface}
        ${MLIROpenMP}
        ${MLIRMlirOptMain}
        ${MLIRSideEffects}        
        ${MLIRStandardOps}
        ${MLIRStandardToLLVM}
        ${MLIRSupport}
        ${MLIRTranslation}
        # strict order verified
        ${LLVMBitWriter}
        ${LLVMObject}
        ${LLVMBitReader}
        # strict order verified
        ${LLVMFrontendOpenMP}
        ${LLVMTransformUtils}
        ${LLVMAnalysis}
        # strict order verified
        ${LLVMAsmParser}
        ${LLVMCore}
        # strict order not yet verified
        ${LLVMRemarks}
        ${LLVMMCParser}
        ${LLVMMC}
        ${LLVMProfileData}
        ${LLVMBinaryFormat}
        ${LLVMBitstreamReader}
        ${LLVMIRReader}
        ${LLVMMLIRTableGen}
        ${LLVMSupport}
        ${LLVMDemangle}
        ${CMAKE_THREAD_LIBS_INIT}
	${CURSES_LIBRARIES}
	${ZLIB_LIBRARIES})

# MLIR libraries that must be linked with --whole-archive for static build or
# must be specified on LD_PRELOAD for shared build.
set(MLIRWholeArchiveLibs
        MLIRAffineToStandard
        MLIRAffine
        MLIRLLVMIR
        MLIRStandardOps
        MLIRStandardToLLVM
        MLIRTransforms
        MLIRLoopToStandard
        MLIRVector
        MLIRLoopOps)

# ONNX MLIR libraries that must be linked with --whole-archive for static build or
# must be specified on LD_PRELOAD for shared build.
set(ONNXMLIRWholeArchiveLibs
        OMKrnlToAffine
        OMKrnlToLLVM
        OMONNXToKrnl
        OMONNXRewrite
        OMShapeInference
        OMShapeInferenceOpInterface
        OMAttributePromotion
        OMPromotableConstOperandsOpInterface)

# Function to construct linkage option for the static libraries that must be
# linked with --whole-archive (or equivalent).
function(whole_archive_link target lib_dir)
  get_property(link_flags TARGET ${target} PROPERTY LINK_FLAGS)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(link_flags "${link_flags} -L${lib_dir}  ")
    foreach(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags}
              "-Wl,-force_load, ${lib_dir}/lib${LIB}.a ")
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

# Function to construct LD_PRELOAD value for the shared libraries whose
# static counterpart need --whole-archive linkage option.
function(ld_preload_libs target lib_dir)
  foreach(lib ${ARGN})
    if("${${lib}}" STREQUAL "")
      set(ONNX_MLIR_LD_PRELOAD_${target}
	"${ONNX_MLIR_LD_PRELOAD_${target}}:${lib_dir}/lib${lib}.so"
	CACHE STRING "" FORCE)
    else()
      set(ONNX_MLIR_LD_PRELOAD_${target}
	"${ONNX_MLIR_LD_PRELOAD_${target}}:${${lib}}"
	CACHE STRING "" FORCE)
    endif()
  endforeach(lib)
endfunction(ld_preload_libs)

function(whole_archive_link_mlir target)
  if(BUILD_SHARED_LIBS)
    ld_preload_libs(${target} ${LLVM_PROJ_BUILD}/lib ${ARGN})
  else()
    whole_archive_link(${target} ${LLVM_PROJ_BUILD}/lib ${ARGN})
  endif()
endfunction(whole_archive_link_mlir)

function(whole_archive_link_onnx_mlir target)
  foreach(lib_target ${ARGN})
    add_dependencies(${target} ${lib_target})
  endforeach(lib_target)
  if(BUILD_SHARED_LIBS)
    ld_preload_libs(${target} ${CMAKE_BINARY_DIR}/lib ${ARGN})
  else()
    whole_archive_link(${target} ${CMAKE_BINARY_DIR}/lib ${ARGN})
  endif()
endfunction(whole_archive_link_onnx_mlir)

set(LLVM_CMAKE_DIR
        "${LLVM_PROJ_BUILD}/lib/cmake/llvm"
        CACHE PATH "Path to LLVM cmake modules")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)

function(onnx_mlir_tablegen ofn)
  tablegen(MLIR
          ${ARGV}
          "-I${MLIR_SRC_INCLUDE_PATH}"
          "-I${MLIR_BIN_INCLUDE_PATH}"
          "-I${ONNX_MLIR_SRC_ROOT}")
  set(TABLEGEN_OUTPUT
          ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
          PARENT_SCOPE)
endfunction()

# Import the pre-built mlir TableGen as an imported exetuable. It is required by
# the LLVM TableGen command to have the TableGen target so that changes to the
# table gen utility itself can be detected and cause re-compilation of .td file.
add_executable(mlir-tblgen IMPORTED)
set_property(TARGET mlir-tblgen
        PROPERTY IMPORTED_LOCATION ${LLVM_PROJ_BUILD}/bin/mlir-tblgen)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

# Add a dialect used by ONNX MLIR and copy the generated operation
# documentation to the desired places.
# c.f. https://github.com/llvm/llvm-project/blob/e298e216501abf38b44e690d2b28fc788ffc96cf/mlir/CMakeLists.txt#L11
function(add_onnx_mlir_dialect_doc dialect dialect_tablegen_file)
  # Generate Dialect Documentation
  set(LLVM_TARGET_DEFINITIONS ${dialect_tablegen_file})
  onnx_mlir_tablegen(${dialect}.md -gen-op-doc)
  set(GEN_DOC_FILE ${ONNX_MLIR_BIN_ROOT}/doc/Dialects/${dialect}.md)
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
