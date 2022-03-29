# SPDX-License-Identifier: Apache-2.0

set(ACCEL_LIST "")

# Generate the enumerators used to identify the kind of accelerators.
set(AcceleratorKinds "${CMAKE_CURRENT_BINARY_DIR}/AcceleratorKinds.inc")
file(WRITE "${AcceleratorKinds}" "// Enumeration of accelerators.\n")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
    list(APPEND ACCEL_LIST "${t}Accel")
    file(APPEND "${AcceleratorKinds}" "   ${t},\n")
    endforeach(t)
endif (ACCELERATORS_TO_BUILD)
file(APPEND "${AcceleratorKinds}" "   NONE\n")

# Generate the enumerators for the --maccel command line option.
set(AcceleratorOptions "${CMAKE_CURRENT_BINARY_DIR}/AcceleratorOptions.inc")
file(REMOVE  "${AcceleratorOptions}")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    file(APPEND "${AcceleratorOptions}" "  clEnumValN(accel::Accelerator::Kind::${t}, \"${t}\", \"${t} accelerator\"),\n")
  endforeach(t)  
endif (ACCELERATORS_TO_BUILD)
file(APPEND "${AcceleratorOptions}" "  clEnumValN(accel::Accelerator::Kind::NONE, \"NONE\", \"No accelerator\")\n")

# Generate the function used to initialize the accelerators.
set(InitAccelerators "${CMAKE_CURRENT_BINARY_DIR}/InitAccelerators.cpp")
file(WRITE "${InitAccelerators}" "namespace onnx_mlir {\n"
                                 "namespace accel {\n"
                                 "bool initAccelerators() {\n")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    file(APPEND "${InitAccelerators}" "  extern void create${t}();\n")
    file(APPEND "${InitAccelerators}" "  create${t}();\n")
  endforeach(t)
  file(APPEND "${InitAccelerators}" "  return true;\n")
else (ACCELERATORS_TO_BUILD)
  file(APPEND "${InitAccelerators}" "  return false;\n")
endif (ACCELERATORS_TO_BUILD)
file(APPEND "${InitAccelerators}" "}\n"
                                  "} // namespace accel\n"
                                  "} // namespace onnx_mlir\n")

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
