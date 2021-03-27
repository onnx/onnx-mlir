# SPDX-License-Identifier: Apache-2.0

add_executable(binary-decoder BinaryDecoder.cpp)
target_link_libraries(binary-decoder
        LLVMSupport
        LLVMDemangle
        ${MLIR_SYSTEM_LIBS})

message(STATUS "incl dir" ${ONNX_INCLUDE_DIRS})
target_link_libraries(binary-decoder onnx)