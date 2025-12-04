git fetch --prune --unshallow --tags
sed -i -e 's/target_link_libraries(onnx PUBLIC onnx_proto)/target_link_libraries(onnx PUBLIC onnx_proto PUBLIC ${protobuf_ABSL_USED_TARGETS})/g' CMakeLists.txt
sed -i -e '/absl::log_initialize/a \
          absl::log_internal_check_op\
          absl::log_internal_message\
          absl::log_internal_nullguard\
' CMakeLists.txt
python3 -m pip install .
rm -rf ${HOME}/.cache
