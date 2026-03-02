# SPDX-License-Identifier: Apache-2.0

# Check whether the build is for a tool (sub-project for now)
# Otherwise, continue with default onnx-mlir project
choose_build_target()

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

# Ensure Python bindings are built with onnx-mlir
# since such binding are now necessary to run, for 
# example, RunONNXModel.py.
add_dependencies(onnx-mlir PyRuntimeC PyOMCompileC)
  
