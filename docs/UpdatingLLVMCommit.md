<!--- SPDX-License-Identifier: Apache-2.0 -->

# Updating the LLVM commit or MLIR-HLO submodule

ONNX-MLIR depends on `llvm-project` (among various other projects such as `mlir-hlo`). The `llvm-project` dependency is captured in [utils/clone-mlir.sh](clone-mlir.sh). `mlir-hlo` is a submodule found in the `third_party` directory.

We plan to update `llvm-project` a couple of times a month in order to keep up-to-date with the advancements made in `mlir`, but also to decrease the complexity of each update. There is currently no plan to update `mlir-hlo` on any given schedule, though for a specific LLVM update it may be necessary to also update the `mlir-hlo` submodule for the build to continue working correctly. This is because `mlir-hlo` itself also has a dependency on `mlir`.

## Which LLVM commit should I pick?

Since downstream projects may want to build ONNX-MLIR (and thus LLVM and MLIR-HLO) in various configurations (Release versus Debug builds; on Linux, Windows, or macOS; possibly with Clang, LLD, and LLDB enabled), it is crucial to pick LLVM commits that pass tests for all combinations of these configurations.

Rather than picking independent LLVM commits from other `mlir`-related projects, we leverage the _green_ commits identified by `Torch-MLIR`. These are updated weekly in the following Issue in the `Torch-MLIR` github project: https://github.com/llvm/torch-mlir/issues/1178.

We've started an update rotation that is described [here](https://github.com/onnx/onnx-mlir/wiki/LLVM-Update-Schedule).

## What is the update process?

1. **Lookup green commit hashes**: From the Github issue https://github.com/llvm/torch-mlir/issues/1178, find the LLVM and MLIR-HLO green commits for the week when ONNX-MLIR is being updated.
2. **Update the `llvm-project` commit**: Update the LLVM commit referenced in the source tree to the green commit hash for the LLVM project from Step 1. The current locations that need to be updated are [utils/clone-mlir.sh](clone-mlir.sh), [docs/BuildOnLinuxOSX.md](BuildOnLinuxOSX.md) and  [docs/BuildOnWindows.md](BuildOnWindows.md).
3. **Update the `mlir-hlo` submodule**: In the `third-party/mlir-hlo` directory, run `git fetch` followed by `git checkout <mlir-hlo-commit-hash>` (where `<mlir-hlo-commit-hash>` is the green commit hash for the MLIR-HLO project from Step 1).
4. **Rebuild and test ONNX-MLIR**: This might involve fixing various API breakages introduced upstream (they are likely unrelated to what you are working on).  If these fixes are too complex, please file a work-in-progress PR explaining the issues you are running into asking for help so that someone from the community can help.

Here are some examples of PRs updating the LLVM commit and/or MLIR-HLO submodule:

- https://github.com/onnx/onnx-mlir/pull/1905
- https://github.com/onnx/onnx-mlir/pull/1827