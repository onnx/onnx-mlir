# SPDX-License-Identifier: Apache-2.0
# add_dlc_library(name sources...
#   This function (generally) has the same semantic as add_library. In
#   addition is supports the arguments below and it does the following
#   by default (unless an argument overrides this):
#   1. Add the library
#   2. Add the default target_include_directories
#   3. Add the library to a global property DLC_LIBS
#   4. Add an install target for the library
#   EXCLUDE_FROM_OM_LIBS
#     Do not add the library to the DLC_LIBS property.
#   NO_INSTALL
#     Do not add an install target for the library.
#   DEPENDS targets...
#     Same semantics as add_dependencies().
#   INCLUDE_DIRS include_dirs...
#     Same semantics as target_include_directories().
#   LINK_LIBS lib_targets...
#     Same semantics as target_link_libraries().
#   LINK_COMPONENTS llvm_components...
#     Link the specified LLVM components.
#     Note: only one linkage mode can be specified.
#   )
function(add_dlc_library name)
  cmake_parse_arguments(ARG
    "EXCLUDE_FROM_OM_LIBS;NO_INSTALL"
    ""
    "DEPENDS;INCLUDE_DIRS;LINK_LIBS;LINK_COMPONENTS"
    ${ARGN}
    )

  if (NOT ARG_EXCLUDE_FROM_OM_LIBS)
    set_property(GLOBAL APPEND PROPERTY DLC_LIBS ${name})
  endif()

  add_library(${name} ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})

  if (ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()

  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} ${ARG_INCLUDE_DIRS})
  endif()

  target_include_directories(${name}
    PUBLIC
    ${DLC_SRC_ROOT}
    ${DLC_BIN_ROOT}
    ${DLC_ONNX_MLIR_SRC_ROOT}
    ${DLC_ONNX_MLIR_BIN_ROOT}
    )

  if (ARG_LINK_LIBS)
    target_link_libraries(${name} ${ARG_LINK_LIBS})
  endif()

  if (ARG_LINK_COMPONENTS)
    set(LinkageMode)
    if (ARG_LINK_COMPONENTS MATCHES "^(PUBLIC|PRIVATE|INTERFACE)")
      list(POP_FRONT ARG_LINK_COMPONENTS LinkageMode)
    endif()

    llvm_map_components_to_libnames(COMPONENT_LIBS ${ARG_LINK_COMPONENTS})

    if (LinkageMode)
      target_link_libraries(${name} ${LinkageMode} ${COMPONENT_LIBS})
    else()
      target_link_libraries(${name} PRIVATE ${COMPONENT_LIBS})
    endif()
  endif()

  if (NOT ARG_NO_INSTALL)
    install(TARGETS ${name}
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin
      )
  endif()
endfunction(add_dlc_library)

set(DLC_SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(DLC_BIN_ROOT "${CMAKE_CURRENT_BINARY_DIR}")

set(DLC_LIBRARY_PATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(DLC_RUNTIME_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(DLC_ONNX_MLIR_SRC_ROOT ${ONNX_MLIR_SRC_ROOT})
set(DLC_ONNX_MLIR_BIN_ROOT ${ONNX_MLIR_BIN_ROOT})

# The path to find the zDNN library.
if (DEFINED ENV{ZDNN_LIBRARY_DIR} AND EXISTS $ENV{ZDNN_LIBRARY_DIR})
  message(DEBUG "ZDNN_LIBRARY_DIR        : " $ENV{ZDNN_LIBRARY_DIR})
  set(ZDNN_LIBRARY_DIR $ENV{ZDNN_LIBRARY_DIR})
  set(ZDNN_LIBRARY_ENABLED 1)
else()
  message(DEBUG "ZDNN_LIBRARY_DIR not found. zDNN-related tests will be turned off.")
  set(ZDNN_LIBRARY_ENABLED 0)
endif()

add_subdirectory(Dialect)
add_subdirectory(Conversion)
# add_subdirectory(Pass)
add_subdirectory(Support)
# add_subdirectory(Tool)
add_subdirectory(Transform)
add_subdirectory(Compiler)

add_onnx_mlir_library(NNPAAccel
  NNPAAccelerator.cpp
  ../Accelerator.cpp

  DEPENDS
  DLCompilerUtils
  onnx
  
  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_SRC_ROOT}
  ${DLC_ONNX_MLIR_SRC_ROOT}
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  # /workdir/onnx-mlir/third_party/onnx

  LINK_LIBS PUBLIC
  onnx
  DLCompilerUtils
  OMZHighOps
  OMZLowOps
  )

  target_link_libraries(NNPAAccel PUBLIC DLCompilerUtils)
  target_link_libraries(NNPAAccel PUBLIC OMZHighOps)
  target_link_libraries(NNPAAccel PUBLIC OMZLowOps)

