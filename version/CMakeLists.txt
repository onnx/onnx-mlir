# SPDX-License-Identifier: Apache-2.0

set(llvm_source_dir ${LLVM_DIR})
set(onnx_mlir_source_dir ${ONNX_MLIR_SRC_ROOT})

find_first_existing_vc_file("${onnx_mlir_source_dir}" onnx_mlir_vc)
find_first_existing_vc_file("${llvm_source_dir}" llvm_vc)

set(version_inc "${ONNX_MLIR_BIN_ROOT}/VCSVersion.inc")
set(generate_vcs_version_script "${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake")

add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${llvm_vc}" "${onnx_mlir_vc}" "${generate_vcs_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DNAMES=\"LLVM;ONNX_MLIR\""
                           "-DLLVM_SOURCE_DIR=${llvm_source_dir}"
                           "-DONNX_MLIR_SOURCE_DIR=${onnx_mlir_source_dir}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${generate_vcs_version_script}")

set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

add_custom_target(Version ALL DEPENDS ${version_inc})
add_dependencies(CompilerUtils Version)
