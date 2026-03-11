#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT/utils"

# Run gen_onnx_mlir.py and copy auto-generated files (keep stderr visible for diagnostics)
bash -e gen_onnx_mlir_multiple_custom_ops.sh > /dev/null

# Verify the generator actually produced output
if ! ls *.td 1>/dev/null 2>&1; then
  echo "::error::gen_onnx_mlir_multiple_custom_ops.sh did not produce any .td files — generation likely failed."
  exit 1
fi

# Build the exact list of destination files (mirroring gen_onnx_mlir_multiple_custom_ops.sh copy logic)
generated_files=()

generated_files+=("src/Builder/OpBuildTable.inc")
generated_files+=("src/Dialect/ONNX/ONNXOps.td.inc")

for td_file in *.td; do
  generated_files+=("src/Dialect/ONNX/$td_file")
done

prefixes=()
for td_file in *Ops.td; do
  prefixes+=("${td_file%Ops.td}")
done

cpp_hpp_files=()
for prefix in "${prefixes[@]}"; do
  for f in ${prefix}*.cpp ${prefix}*.hpp; do
    case "$f" in
      *ShapeInference.cpp|*Verify.cpp) continue ;;
    esac
    generated_files+=("src/Dialect/ONNX/ONNXOps/Additional/$f")
    cpp_hpp_files+=("src/Dialect/ONNX/ONNXOps/Additional/$f")
  done
done

cd "$REPO_ROOT"

# Run clang-format only on the generated C++ files
if [ ${#cpp_hpp_files[@]} -gt 0 ]; then
  if command -v clang-format &> /dev/null; then
    clang-format -i "${cpp_hpp_files[@]}"
  else
    echo "::warning::clang-format not found — skipping formatting of generated files."
  fi
fi

# Handle ONNXOps.td.inc separately: allow the known FXML-4138 BF16 manual edit
check_files=()
for f in "${generated_files[@]}"; do
  if [ "$f" = "src/Dialect/ONNX/ONNXOps.td.inc" ]; then
    if ! git diff --quiet -- "$f"; then
      inc_diff=$(git diff -- "$f")
      num_hunks=$(echo "$inc_diff" | grep -c '^@@')
      is_known_bf16_diff=$(echo "$inc_diff" | grep -c 'FIXME(FXML-4138)')
      if [ "$num_hunks" -eq 1 ] && [ "$is_known_bf16_diff" -ge 1 ]; then
        echo "::warning::ONNXOps.td.inc has the expected FXML-4138 BF16 manual edit diff — skipping."
        git checkout -- "$f"
      else
        echo "::error::ONNXOps.td.inc has unexpected differences beyond the known FXML-4138 BF16 edit."
        echo "$inc_diff"
        exit 1
      fi
    fi
  else
    check_files+=("$f")
  fi
done

# Check for diffs only in the generated files
if ! git diff --quiet -- "${check_files[@]}"; then
  echo "::error::Generated files are not in sync with gen_onnx_mlir.py and/or yaml files. Please run utils/gen_onnx_mlir_multiple_custom_ops.sh or refer to docs/ImportONNXDefs.md."
  git diff --stat -- "${check_files[@]}"
  git diff -- "${check_files[@]}"
  exit 1
fi

# Check for untracked files only among the generated files
for f in "${check_files[@]}"; do
  if [ -n "$(git ls-files --others --exclude-standard -- "$f")" ]; then
    echo "::error::Generated file $f is untracked. Please commit it or update .gitignore."
    exit 1
  fi
done
