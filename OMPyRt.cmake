option(ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT "Set to ON for building Python driver of running the compiled model without llvm-project." ON)

# Use the submodule if present, otherwise find pybind11 installed by pip.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt)
  add_subdirectory(third_party/pybind11)
else()
  find_package(pybind11 REQUIRED)
endif()

add_subdirectory(src)
