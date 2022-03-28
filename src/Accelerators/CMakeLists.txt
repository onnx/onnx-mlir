# SPDX-License-Identifier: Apache-2.0
set(ACCEL_LIST "")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/AcceleratorKinds.hpp "// Enumeration of accelerators \n")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
    list(APPEND ACCEL_LIST "${t}Accel")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/AcceleratorKinds.hpp "   ${t},\n")
    endforeach(t)
else (ACCELERATORS_TO_BUILD)
endif (ACCELERATORS_TO_BUILD)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/AcceleratorKinds.hpp "   NONE\n")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "bool InitAccelerators() {\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/AcceleratorOptions.hpp "   clEnumValN(\n")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "extern void create${t}();\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "create${t}();\n")
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/AcceleratorOptions.hpp "      accel::Accelerator::Kind::${t}, \"${t}\", \"Some Accelerator\"),\n    clEnumValN(\n")
  endforeach(t)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "return true;\n")
else (ACCELERATORS_TO_BUILD)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "return false;\n")
endif (ACCELERATORS_TO_BUILD)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "}\n")
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/AcceleratorOptions.hpp "      accel::Accelerator::Kind::NONE, \"NONE\", \"Some Accelerator\")\n")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "bool InitAccelerators() {\n")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "extern void create${t}();\n")
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "create${t}();\n")
  endforeach(t)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "return true;\n")
else (ACCELERATORS_TO_BUILD)
  file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "return false;\n")
endif (ACCELERATORS_TO_BUILD)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp "}\n")

add_onnx_mlir_library(InitAccelerators
  InitAccelerators.cpp

  EXCLUDE_FROM_OM_LIBS

  LINK_LIBS PUBLIC
    ${ACCEL_LIST}
  )

add_onnx_mlir_library(Accelerator
  Accelerator.cpp
  DEPENDS
    MLIRIR
  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
  LINK_LIBS PUBLIC
    LLVMSupport
  )
