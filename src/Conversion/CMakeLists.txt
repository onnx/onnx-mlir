# SPDX-License-Identifier: Apache-2.0

add_subdirectory(ONNXToKrnl)
add_subdirectory(KrnlToLLVM)
add_subdirectory(KrnlToAffine)
add_subdirectory(SeqToMemref)
add_subdirectory(ONNXToTOSA)

if (ONNX_MLIR_ENABLE_MHLO)
  add_subdirectory(ONNXToMhlo)
endif()
