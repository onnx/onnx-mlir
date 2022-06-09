# SPDX-License-Identifier: Apache-2.0

# We would like the VCSVersion.inc file to contain both the llvm and onnx-mlir
# git revision. This is trivial for onnx-mlir, however, we cannot guarantee
# that we will have access to the llvm source. If the source directory exists,
# we attempt to use it to get the version. Otherwise, we rely on LLVM_APPEND_VC_REV
# and VCSRevision.h.

set(version_inc "${ONNX_MLIR_BIN_ROOT}/VCSVersion.inc")
set(generate_vcs_version_script "${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake")
set(vcs_depends "${generate_vcs_version_script}")

find_first_existing_vc_file("${ONNX_MLIR_SRC_ROOT}" onnx_mlir_vc)
list(APPEND vcs_depends "${onnx_mlir_vc}")

if (LLVM_APPEND_VC_REV)
  set(onnx_mlir_source_dir ${ONNX_MLIR_SRC_ROOT})
  set(llvm_source_dir ${LLVM_BUILD_MAIN_SRC_DIR})
endif()

list(APPEND vcs_generate_commands
  COMMAND ${CMAKE_COMMAND} [[-DNAMES="LLVM\;ONNX_MLIR"]]
                           "-DLLVM_SOURCE_DIR=${llvm_source_dir}"
                           "-DONNX_MLIR_SOURCE_DIR=${onnx_mlir_source_dir}"
                           "-DHEADER_FILE=${version_inc}.tmp"
                           -P "${generate_vcs_version_script}"
  )

find_first_existing_vc_file("${LLVM_BUILD_MAIN_SRC_DIR}" llvm_vc)
if (llvm_vc)
  list(APPEND vcs_depends "${llvm_vc}")
elseif (LLVM_APPEND_VC_REV)
  set(add_llvm_version_header_script "${CMAKE_CURRENT_SOURCE_DIR}/AddLLVMVersionHeader.cmake")
  list(APPEND vcs_depends "${add_llvm_version_header_script}")
  list(APPEND vcs_generate_commands
    COMMAND ${CMAKE_COMMAND} "-DHEADER_FILE=${version_inc}.tmp"
                             -P "${add_llvm_version_header_script}"
    )
endif()

list(APPEND vcs_generate_commands
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${version_inc}.tmp" "${version_inc}"
  COMMAND ${CMAKE_COMMAND} -E remove "${version_inc}.tmp"
  )

add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${vcs_depends}"
  ${vcs_generate_commands}
  )

set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE
  )

add_onnx_mlir_library(OMVersion
  Version.cpp
  ${version_inc}

  LINK_LIBS PUBLIC
    LLVMSupport
)

file(READ "${ONNX_MLIR_SRC_ROOT}/VERSION_NUMBER" ONNX_MLIR_VERSION)
string(STRIP "${ONNX_MLIR_VERSION}" ONNX_MLIR_VERSION)
list(APPEND DEFINITIONS "ONNX_MLIR_VERSION=\"${ONNX_MLIR_VERSION}\"")
file(READ "${ONNX_MLIR_SRC_ROOT}/third_party/onnx/VERSION_NUMBER" ONNX_VERSION)
string(STRIP "${ONNX_VERSION}" ONNX_VERSION)
if ("${ONNX_VERSION}" STREQUAL "")
  message(WARNING "third_party/onnx version not detected, you may be on the wrong commit hash")
else()
  message(STATUS "Onnx version             : ${ONNX_VERSION}")
  list(APPEND DEFINITIONS "ONNX_VERSION=\"${ONNX_VERSION}\"")
endif()
if (ONNX_MLIR_VENDOR)
  list(APPEND DEFINITIONS "ONNX_MLIR_VENDOR=\"${ONNX_MLIR_VENDOR }\"")
endif()
list(APPEND DEFINITIONS "LLVM_PACKAGE_VERSION=\"${LLVM_PACKAGE_VERSION}\"")
target_compile_definitions(OMVersion PUBLIC ${DEFINITIONS})
