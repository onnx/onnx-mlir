# SPDX-License-Identifier: Apache-2.0

if (ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT)
  add_subdirectory(Runtime)
  # Accelerators introduces a target AcceleratorsInc. Define a dummy one here
  add_custom_target(AcceleratorsInc
      COMMAND echo "This is the dummy definition for AcceleratorsInc"
  ) 
  add_compile_definitions(ENABLE_PYRUNTIME_LIGHT)
  return()
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
  
