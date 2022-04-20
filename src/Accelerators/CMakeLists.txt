# SPDX-License-Identifier: Apache-2.0

# Populate the accelerator list and add the accelerator subdirectories.
set(ACCEL_LIST "")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
    list(APPEND ACCEL_LIST "${t}Accel")
  endforeach(t)
  message(STATUS "Accelerators             : ${ACCEL_LIST}")
else()
  message(STATUS "Accelerators             : NONE")
endif (ACCELERATORS_TO_BUILD)

# Generate the Accelerators.inc file using a cmake script.
set(accelerators_inc "${CMAKE_CURRENT_BINARY_DIR}/Accelerators.inc")
set(create_accelerators_inc_script "${CMAKE_CURRENT_SOURCE_DIR}/CreateAcceleratorsInc.cmake")

add_custom_command(OUTPUT "${accelerators_inc}"
  DEPENDS "${create_accelerators_inc_script}"
  COMMAND ${CMAKE_COMMAND} "-DINC_FILE=${accelerators_inc}" "-DACCELERATORS=${ACCELERATORS_TO_BUILD}" 
    -P "${create_accelerators_inc_script}"
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
