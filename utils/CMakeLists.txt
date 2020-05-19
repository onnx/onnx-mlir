# Invoke gen_doc.py to obtain ONNXOps.td.inc, OpBuildTable.inc.
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/ONNXOps.td.inc
                ${CMAKE_CURRENT_SOURCE_DIR}/OpBuildTable.inc
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/gen_doc.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_doc.py)

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

# Invoke gen_doc.py to obtain ONNXOps.td.inc, OpBuildTable.inc.
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/MLONNXOps.td.inc
                ${CMAKE_CURRENT_SOURCE_DIR}/MLOpBuildTable.inc
       	COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/gen_doc.py --domain="ONNX_ML"
       	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_doc.py)

# Move the generated files to respective destinations:
# MLONNXOps.td.inc -> src/Dialect/MLONNX/MLONNXOps.td.inc
add_custom_target(OMMLONNXOpsTableGenIncGen
       	COMMAND ${CMAKE_COMMAND} -E rename
               	${CMAKE_CURRENT_SOURCE_DIR}/MLONNXOps.td.inc
               	${ONNX_MLIR_SRC_ROOT}/src/Dialect/MLONNX/MLONNXOps.td.inc
       	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/MLONNXOps.td.inc)

# MLOpBuildTable.inc -> src/Builder/MLOpBuildTable.inc
add_custom_target(OMMLONNXOpsBuildTableIncGen
       	COMMAND ${CMAKE_COMMAND} -E rename
       	${CMAKE_CURRENT_SOURCE_DIR}/MLOpBuildTable.inc
       	${ONNX_MLIR_SRC_ROOT}/src/Builder/MLOpBuildTable.inc
       	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/MLOpBuildTable.inc)

add_custom_target(OMMLONNXOpsIncTranslation
       	DEPENDS OMMLONNXOpsTableGenIncGen
               	OMMLONNXOpsBuildTableIncGen)
