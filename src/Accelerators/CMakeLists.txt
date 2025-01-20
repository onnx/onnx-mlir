# SPDX-License-Identifier: Apache-2.0

# Populate the accelerator list and add the accelerator subdirectories.
# ONNX_MLIR_ACCELERATORS is the semicolon-separated list of accelerators user specified
# Note that the list should be quoted, e.g. -DONNX_MLIR_ACCELERATORS='A;B'
# ACCEL_TARGET_LIST is the list of cmake targets
# ACCEL_LINK_LIST is the lists of accelerator libraries
# ACCEL_INCLUDE_LIST is the list passed to inc generator
if (ONNX_MLIR_ACCELERATORS)
  foreach(t ${ONNX_MLIR_ACCELERATORS})
    add_subdirectory(${t})

    # If the accelerator can be built
    string(TOUPPER ${t} T)
    if (${T}_ENABLED)
      list(APPEND ACCEL_TARGET_LIST "${t}Accel")
      list(APPEND ACCEL_LINK_LIST "OM${t}Accel")
      list(APPEND ACCEL_INCLUDE_LIST "${t}")
    endif()
  endforeach(t)
endif (ONNX_MLIR_ACCELERATORS)
if (ACCEL_INCLUDE_LIST)
  message(STATUS "Accelerators             : ${ACCEL_INCLUDE_LIST}")
else()
  message(STATUS "Accelerators             : NONE")
endif()

# Generate the Accelerators.inc file using a cmake script.
set(accelerators_inc "${CMAKE_CURRENT_BINARY_DIR}/Accelerators.inc")
set(create_accelerators_inc_script "${CMAKE_CURRENT_SOURCE_DIR}/CreateAcceleratorsInc.cmake")

add_custom_command(OUTPUT "${accelerators_inc}"
  DEPENDS "${create_accelerators_inc_script}"
  COMMAND ${CMAKE_COMMAND} "-DINC_FILE=${accelerators_inc}" "-DACCELERATORS=\"${ACCEL_INCLUDE_LIST}\""
    -P "${create_accelerators_inc_script}"
  )
set_source_files_properties("${accelerators_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE
  )
add_custom_target(AcceleratorsInc ALL DEPENDS ${accelerators_inc})

add_onnx_mlir_library(OMInitAccelerators
  InitAccelerators.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS 
    AcceleratorsInc    

  LINK_LIBS PUBLIC
    ${ACCEL_LINK_LIST}
    LLVMSupport
    MLIRIR
  )

add_onnx_mlir_library(OMAccelerator
  Accelerator.cpp

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
    AcceleratorsInc    

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
    LLVMSupport
    MLIRIR
  )
