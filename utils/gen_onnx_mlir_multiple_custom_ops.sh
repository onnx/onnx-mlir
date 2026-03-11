# Generate ONNX MLIR op files for multiple custom ops
# Loop through yaml files in current directory to create an argument list to gen_onnx_mlir.py
yaml_args=()
for yaml_file in *.yaml; do
    yaml_args+=(--custom-ops-yaml "$yaml_file")
done
# print yaml_args
echo "yaml_args: ${yaml_args[@]}"
python gen_onnx_mlir.py "${yaml_args[@]}"

echo "========================================================="
echo "Copying OpBuildTable.inc to ../src/Builder/"
cp OpBuildTable.inc ../src/Builder/

echo "========================================================="
echo "Copying ONNXOps.td.inc to ../src/Dialect/ONNX/"
cp ONNXOps.td.inc ../src/Dialect/ONNX/
echo "WARNING: ONNXOps.td.inc has manual edits which have been overwritten by the script. Please check before committing."



echo "========================================================="
echo "Copying generated .td files to ../src/Dialect/ONNX/"
ls *.td
cp *.td ../src/Dialect/ONNX/
echo "========================================================="

# extract prefix from .td file names (until <prefix>Ops.td)
prefixes=()
for td_file in *Ops.td; do
    prefix="${td_file%Ops.td}"
    prefixes+=("$prefix")
done
echo "Prefixes: ${prefixes[@]}"

echo "Copying generated <prefix>*.cpp/.hpp files (excluding shape inference and verify cpp files) to ../src/Dialect/ONNX/ONNXOps/Additional/"
for prefix in "${prefixes[@]}"; do
    ls ${prefix}*.cpp ${prefix}*.hpp | grep -v "ShapeInference.cpp" | grep -v "Verify.cpp" | while read -r f; do
        cp "$f" ../src/Dialect/ONNX/ONNXOps/Additional/
    done
done
echo "========================================================="
echo "REMINDER: Add your function implementation (for a new op only) to files:"
ls *ShapeInference.cpp *Verify.cpp
echo "in ../src/Dialect/ONNX/ONNXOps/Additional/"
echo "========================================================="