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

# Copy LLVM tools (llc and opt) to onnx-mlir binary directory in standalone mode.
if(ONNX_MLIR_BUILD_STANDALONE)
  # Determine the executable extension for the platform.
  if(WIN32)
    set(LLVM_TOOL_EXT ".exe")
  else()
    set(LLVM_TOOL_EXT "")
  endif()

  # Derive LLVM tools binary directory from LLVM_DIR.
  # LLVM_DIR is typically <llvm-build>/lib/cmake/llvm, so we go up 3 levels to get to <llvm-build>.
  get_filename_component(LLVM_BUILD_DIR "${LLVM_DIR}/../../.." ABSOLUTE)
  set(LLVM_TOOLS_BIN_DIR "${LLVM_BUILD_DIR}/bin")

  # Define source paths for llc and opt.
  set(LLC_SOURCE "${LLVM_TOOLS_BIN_DIR}/llc${LLVM_TOOL_EXT}")
  set(OPT_SOURCE "${LLVM_TOOLS_BIN_DIR}/opt${LLVM_TOOL_EXT}")

  # Verify that the tools exist.
  if(EXISTS "${LLC_SOURCE}" AND EXISTS "${OPT_SOURCE}")
    # Add custom commands to copy llc and opt after building onnx-mlir.
    add_custom_command(TARGET onnx-mlir POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLC_SOURCE}"
        "$<TARGET_FILE_DIR:onnx-mlir>/llc${LLVM_TOOL_EXT}"
      COMMENT "Copying llc to onnx-mlir binary directory"
      VERBATIM
    )

    add_custom_command(TARGET onnx-mlir POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPT_SOURCE}"
        "$<TARGET_FILE_DIR:onnx-mlir>/opt${LLVM_TOOL_EXT}"
      COMMENT "Copying opt to onnx-mlir binary directory"
      VERBATIM
    )

    message(STATUS "ONNX_MLIR_BUILD_STANDALONE: Will copy llc and opt from ${LLVM_TOOLS_BIN_DIR}")
  else()
    message(WARNING "ONNX_MLIR_BUILD_STANDALONE is ON but llc or opt not found in ${LLVM_TOOLS_BIN_DIR}")
  endif()
endif()
  
