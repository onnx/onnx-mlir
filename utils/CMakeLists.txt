# SPDX-License-Identifier: Apache-2.0

# Invoke gen_onnx_mlir.py to obtain ONNXOps.td.inc, OpBuildTable.inc.
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/ONNXOps.td.inc
        ${CMAKE_CURRENT_SOURCE_DIR}/OpBuildTable.inc
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_LIST_DIR}/gen_onnx_mlir.py
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/gen_onnx_mlir.py)

# Move the generated files to respective destinations:
# ONNXOps.td.inc -> src/Dialect/ONNX/ONNXOps.td.inc
add_custom_target(OMONNXOpsTableGenIncGen
        COMMAND ${CMAKE_COMMAND} -E rename
        ${CMAKE_CURRENT_SOURCE_DIR}/ONNXOps.td.inc
        ${ONNX_MLIR_SRC_ROOT}/src/Dialect/ONNX/ONNXOps.td.inc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ONNXOps.td.inc)

# OpBuildTable.inc -> src/Builder/OpBuildTable.inc
add_custom_target(OMONNXOpsBuildTableIncGen
        COMMAND ${CMAKE_COMMAND} -E rename
        ${CMAKE_CURRENT_SOURCE_DIR}/OpBuildTable.inc
        ${ONNX_MLIR_SRC_ROOT}/src/Builder/OpBuildTable.inc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/OpBuildTable.inc)

add_custom_target(OMONNXOpsIncTranslation
        DEPENDS OMONNXOpsTableGenIncGen
        OMONNXOpsBuildTableIncGen)

add_custom_target(OMONNXCheckVersion
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/gen_onnx_mlir.py --check-operation-version)


# Scan files for supported ops
add_onnx_mlir_supported_ops(${ONNX_MLIR_SRC_ROOT}/test/backend/inference_backend.py cpu)
add_onnx_mlir_supported_ops(${ONNX_MLIR_SRC_ROOT}/test/accelerators/NNPA/backend/CMakeLists.txt NNPA)

add_custom_target(OMCreateONNXMLIRSource
        COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/onnxmlir ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/RunONNXModel.py ${CMAKE_CURRENT_BINARY_DIR}/onnxmlir/src/onnxmlir)

