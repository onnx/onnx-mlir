git fetch --prune --unshallow --tags
sed -i -e 's/target_link_libraries(onnx PUBLIC onnx_proto)/target_link_libraries(onnx PUBLIC onnx_proto PRIVATE ${protobuf_ABSL_USED_TARGETS})/g' \
       -e '/absl::log_initialize/a \
          absl::log_internal_check_op\
          absl::log_internal_message\
          absl::log_internal_nullguard' CMakeLists.txt
python3 -m pip install -v .
rm -rf ${HOME}/.cache
