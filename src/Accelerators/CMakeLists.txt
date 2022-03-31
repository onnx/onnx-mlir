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

# Generate the Accelerators.inc file.
set(Accelerators "${CMAKE_CURRENT_BINARY_DIR}/Accelerators.inc")
if (ACCELERATORS_TO_BUILD) 
  file(WRITE "${Accelerators}" "#define HAS_ACCELERATORS\n"
                               "#define APPLY_TO_ACCELERATORS(MACRO) \\\n")
  foreach(t ${ACCELERATORS_TO_BUILD})
    file(APPEND "${Accelerators}" "  MACRO(${t})\\\n")
  endforeach(t)
else (ACCELERATORS_TO_BUILD)
  file(WRITE "${Accelerators}" "#define APPLY_TO_ACCELERATORS(MACRO) \\\n")
endif (ACCELERATORS_TO_BUILD)

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
