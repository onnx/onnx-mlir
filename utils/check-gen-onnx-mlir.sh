#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT/utils"

# Run gen_onnx_mlir.py and copy auto-generated files
bash gen_onnx_mlir_multiple_custom_ops.sh > /dev/null 2>&1

cd "$REPO_ROOT"

# Run clang-format only on the C++ files produced by the generator
find src/Dialect/ONNX/ONNXOps/Additional -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

# Check for expected manual edits in ONNXOps.td.inc
if ! git diff --quiet -- src/Dialect/ONNX/ONNXOps.td.inc; then
  inc_diff=$(git diff -- src/Dialect/ONNX/ONNXOps.td.inc)
  num_hunks=$(echo "$inc_diff" | grep -c '^@@')
  is_known_bf16_diff=$(echo "$inc_diff" | grep -c 'FIXME(FXML-4138)')
  if [ "$num_hunks" -eq 1 ] && [ "$is_known_bf16_diff" -ge 1 ]; then
    echo "::warning::ONNXOps.td.inc has the expected FXML-4138 BF16 manual edit diff — skipping."
    git checkout -- src/Dialect/ONNX/ONNXOps.td.inc
  else
    echo "::error::ONNXOps.td.inc has unexpected differences beyond the known FXML-4138 BF16 edit."
    echo "$inc_diff"
    exit 1
  fi
fi

# Check for unexpected file changes
if ! git diff --quiet; then
  echo "::error::Generated files are out of date. Please run utils/gen_onnx_mlir_multiple_custom_ops.sh and commit the results."
  git diff --stat
  git diff
  exit 1
fi

# Check for untracked files outside of utils/
untracked=$(git ls-files --others --exclude-standard -- ':!utils/')
if [ -n "$untracked" ]; then
  echo "::error::gen_onnx_mlir_multiple_custom_ops.sh produced untracked files. Please commit them or update .gitignore."
  echo "$untracked"
  exit 1
fi
