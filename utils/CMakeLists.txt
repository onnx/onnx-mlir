# Invoke gen_doc.py to obtain ONNXOps.td.inc, OpBuildTable.inc.
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/ONNXOps.td.inc
                ${CMAKE_CURRENT_SOURCE_DIR}/OpBuildTable.inc
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/gen_doc.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_doc.py)

# Copy the generated files to respective destinations:
# ONNXOps.td.inc -> src/Dialect/ONNX/ONNXOps.td.inc
# OpBuildTable.inc -> src/Builder/OpBuildTable.inc
add_custom_command(OUTPUT ${ONNX_MLIR_SRC_ROOT}/src/Dialect/ONNX/ONNXOps.td.inc
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/ONNXOps.td.inc
                ${ONNX_MLIR_SRC_ROOT}/src/Dialect/ONNX/ONNXOps.td.inc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ONNXOps.td.inc)
add_custom_command(OUTPUT ${ONNX_MLIR_SRC_ROOT}/src/builder/OpBuildTable.inc
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/OpBuildTable.inc
        ${ONNX_MLIR_SRC_ROOT}/src/Builder/OpBuildTable.inc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/OpBuildTable.inc)

add_custom_target(OMONNXOpsTableGenIncGen
        DEPENDS ${ONNX_MLIR_SRC_ROOT}/src/Dialect/ONNX/ONNXOps.td.inc)
add_custom_target(OMONNXOpsBuildTableIncGen
        DEPENDS ${ONNX_MLIR_SRC_ROOT}/src/builder/OpBuildTable.inc)