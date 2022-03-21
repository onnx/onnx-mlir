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

list(APPEND vcs_generate_commands
  COMMAND ${CMAKE_COMMAND} [[-DNAMES="LLVM\;ONNX_MLIR"]]
                           "-DLLVM_SOURCE_DIR=${LLVM_BUILD_MAIN_SRC_DIR}"
                           "-DONNX_MLIR_SOURCE_DIR=${ONNX_MLIR_SRC_ROOT}"
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

add_custom_target(OMVersion ALL DEPENDS ${version_inc})
