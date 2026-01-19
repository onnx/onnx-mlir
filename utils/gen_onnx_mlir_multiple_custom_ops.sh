# Generate ONNX MLIR op files for multiple custom ops
# Loop through yaml files in current directory to create an argument list to gen_onnx_mlir.py
yaml_args=()
for yaml_file in *.yaml; do
    yaml_args+=(--custom-ops-yaml "$yaml_file")
done
# print yaml_args
echo "yaml_args: ${yaml_args[@]}"
python gen_onnx_mlir.py "${yaml_args[@]}"