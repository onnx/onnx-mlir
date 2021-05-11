add_custom_target(copy-maia
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/maia.py ${ONNX_MLIR_RUNTIME_PATH}
    COMMENT "Copying maia.py"
)