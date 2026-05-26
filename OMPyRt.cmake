option(ONNX_MLIR_ENABLE_PYRUNTIME_LIGHT "Set to ON for building Python driver of running the compiled model without llvm-project." ON)
add_subdirectory(third_party/pybind11)
add_subdirectory(src)
