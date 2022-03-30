# SPDX-License-Identifier: Apache-2.0

# Populate the accelerator list and add the accelerator subdirectories.
set(ACCEL_LIST "")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
    list(APPEND ACCEL_LIST "${t}Accel")
  endforeach(t)
endif (ACCELERATORS_TO_BUILD)

# Generate the enumerators used to identify the kind of accelerators.
set(AcceleratorKinds "${CMAKE_CURRENT_BINARY_DIR}/AcceleratorKinds.inc")
file(WRITE "${AcceleratorKinds}" "#if !defined(ACCEL_KIND)\n"
                                 "#error \"Must define ACCEL_KIND macro.\"\n"
                                 "#else\n")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    file(APPEND "${AcceleratorKinds}" "  ACCEL_KIND(${t})\n")
  endforeach(t)
endif (ACCELERATORS_TO_BUILD)
file(APPEND "${AcceleratorKinds}" "#endif\n")

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

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
    MLIRIR

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
    LLVMSupport
  )
