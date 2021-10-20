if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Suppress warnings in third party subprojects.
  add_compile_options(-Wno-everything)
endif()

add_subdirectory(onnx)
add_subdirectory(pybind11)
add_subdirectory(rapidcheck)
