# SPDX-License-Identifier: Apache-2.0

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

if (ONNX_MLIR_BUILD_SHARED)
  # Gather all mlir library exports
  get_property(_om_targets GLOBAL PROPERTY ONNX_MLIR_LIBS)
  foreach(_target IN LISTS _om_targets)
    list(APPEND _om_objects $<TARGET_OBJECTS:${_target}>)
  endforeach()
  # Append non mlir library exports
  list(APPEND _om_objects $<TARGET_OBJECTS:OMAccelerator>)
  list(APPEND _om_objects $<TARGET_OBJECTS:OMCompiler>)
  list(APPEND _om_objects $<TARGET_OBJECTS:OMCompilerOptions>)
  list(APPEND _om_objects $<TARGET_OBJECTS:OMCompilerPasses>)
  list(APPEND _om_objects $<TARGET_OBJECTS:OMCompilerUtils>)
  list(APPEND _om_objects $<TARGET_OBJECTS:OMInitAccelerators>)
  list(APPEND _om_objects $<TARGET_OBJECTS:OMOptionUtils>)

  # Single shared library
  add_library(ONNXMLIR SHARED
    ${_om_objects}
  )
  target_link_libraries(ONNXMLIR PUBLIC MLIR LLVM onnx onnx_proto)
  install(TARGETS ONNXMLIR LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

add_onnx_mlir_executable(onnx-mlir
  onnx-mlir.cpp

  LINK_LIBS PRIVATE
  MLIROpenMPToLLVMIRTranslation
  OMCompilerOptions
  OMCompilerUtils
  OMOptionUtils
  )
