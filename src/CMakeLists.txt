# SPDX-License-Identifier: Apache-2.0

if (ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
  add_subdirectory(Runtime)
  add_subdirectory(Support)
  # Accelerators introduces a target AcceleratorsInc. Define a dummy one here
  add_custom_target(AcceleratorsInc
      COMMAND echo "This is the dummy definition for AcceleratorsInc"
  ) 
  return()
endif()

# Find OpenMP headers built by LLVM
find_path(OPENMP_INCLUDE_DIR
  NAMES omp.h
  HINTS
    ${LLVM_BUILD_BINARY_DIR}/lib/clang/*/include
  NO_DEFAULT_PATH
)
if(OPENMP_INCLUDE_DIR)
  set(OMP_SUPPORTED ON)
else()
  set(OMP_SUPPORTED OFF)
endif()

add_subdirectory(Accelerators)
add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Builder)
add_subdirectory(Runtime)
add_subdirectory(Tools)
add_subdirectory(Version)
add_subdirectory(Compiler)

add_onnx_mlir_executable(onnx-mlir
  onnx-mlir.cpp

  LINK_LIBS PRIVATE
  OMCompilerOptions
  OMCompilerUtils
  OMOptionUtils
  )
  
