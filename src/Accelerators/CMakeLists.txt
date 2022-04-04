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
set(accelerators_inc "${CMAKE_CURRENT_BINARY_DIR}/Accelerators.inc")
list(APPEND accelerators_inc_generate_commands
  COMMAND ${CMAKE_COMMAND} -E echo "\\#define APPLY_TO_ACCELERATORS\\(MACRO\\) \\ " > ${accelerators_inc})

if (ACCELERATORS_TO_BUILD) 
  foreach(t ${ACCELERATORS_TO_BUILD})
    list(APPEND accelerators_inc_generate_commands
      COMMAND ${CMAKE_COMMAND} -E echo "  MACRO\\(${t}\\) \\ " >> ${accelerators_inc})
  endforeach(t)
  list(APPEND accelerators_inc_generate_commands
      COMMAND ${CMAKE_COMMAND} -E echo >> ${accelerators_inc})
endif (ACCELERATORS_TO_BUILD)

add_custom_command(OUTPUT "${accelerators_inc}"
  DEPENDS ""
  ${accelerators_inc_generate_commands}
  )
set_source_files_properties("${accelerators_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE
  )
add_custom_target(AcceleratorsInc ALL DEPENDS ${accelerators_inc})

add_onnx_mlir_library(InitAccelerators
  InitAccelerators.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS 
    AcceleratorsInc

  LINK_LIBS PUBLIC
    ${ACCEL_LIST}
    LLVMSupport
  )

add_onnx_mlir_library(Accelerator
  Accelerator.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
    AcceleratorsInc
    MLIRIR

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
    LLVMSupport
  )
