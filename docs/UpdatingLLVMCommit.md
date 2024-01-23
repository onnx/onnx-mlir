<!--- SPDX-License-Identifier: Apache-2.0 -->

# Updating the LLVM commit and StableHLO submodule

ONNX-MLIR depends on `LLVM project` (among various other projects such as `StableHLO`). The `LLVM project` dependency is captured in [../utils/clone-mlir.sh](clone-mlir.sh). `StableHLO` is a submodule found in the `third_party` directory.

We plan to update `LLVM project` and `StableHLO` biweekly in order to keep up-to-date with the advancements made in `mlir`, but also to decrease the complexity of each update.

## Which LLVM commit should I pick?

Since downstream projects may want to build ONNX-MLIR (and thus LLVM and StableHLO) in various configurations (Release versus Debug builds; on Linux, Windows, or macOS; possibly with Clang, LLD, and LLDB enabled), it is crucial to pick LLVM commits that pass tests for all combinations of these configurations.

Rather than picking independent LLVM commits from other `mlir`-related projects, we leverage the commits identified by `StableHLO` which is based on `XLA`. Biweekly, StableHLO will bump the revision of LLVM to match what `openxla/xla` is using.

We've started an update rotation that is described [here](https://github.com/onnx/onnx-mlir/wiki/LLVM-Update-Schedule).

## What is the update process?

1. **Lookup commit hashes**: You can find the LLVM commit [here](https://github.com/openxla/stablehlo/blob/main/build_tools/llvm_version.txt) and the StableHLO commit associated with the LLVM commit [here](https://github.com/openxla/stablehlo/tree/main/build_tools). Please check the PR for LLVM to obtain the appropriate commit for StableHLO, we want to upgrade both dependencies in order to avoid any errors.
2. **Update the `llvm-project` commit**: Update the LLVM commit referenced in the source tree to the commit hash for the LLVM project from Step 1. The current locations that need to be updated are [utils/clone-mlir.sh](../utils/clone-mlir.sh), [docs/BuildOnLinuxOSX.md](BuildOnLinuxOSX.md) and  [docs/BuildOnWindows.md](BuildOnWindows.md).
3. **Update the `stablehlo` submodule**: In the `third-party/stablehlo` directory, run `git fetch` followed by `git checkout <stablehlo-commit-hash>` (where `<stablehlo-commit-hash>` is the commit hash for the  project from Step 1).
4. **Rebuild and test ONNX-MLIR**: This might involve fixing various API breakages introduced upstream (they are likely unrelated to what you are working on).  If these fixes are too complex, please file a work-in-progress PR explaining the issues you are running into asking for help so that someone from the community can help.

Here is an example of a PR updating the LLVM commit and StableHLO submodule:

- https://github.com/onnx/onnx-mlir/pull/2662
