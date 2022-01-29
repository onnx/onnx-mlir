# SPDX-License-Identifier: Apache-2.0

add_subdirectory(Interface)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Transform)
add_subdirectory(Builder)
add_subdirectory(Runtime)
add_subdirectory(Tool)
add_subdirectory(Compiler)


if( ACCELERATORS_TO_BUILD )
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(STATUS "Targeting ${t}")
    add_subdirectory(Accelerators/${t})
    endforeach(t)
else( ACCELERATORS_TO_BUILD )
  message(STATUS "No Accelerators Targeted")
  add_subdirectory(Accelerators/None)
endif( MSVC )


add_onnx_mlir_executable(onnx-mlir
  main.cpp

  LINK_LIBS PRIVATE
  CompilerUtils
  )
