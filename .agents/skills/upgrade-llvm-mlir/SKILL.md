---
name: upgrade-llvm-mlir
description: 'Use when bumping onnx-mlir''s LLVM/MLIR commit (utils/clone-mlir.sh) and the StableHLO submodule: find the target SHAs, assess upstream changes, update pins, rebuild LLVM, fix API breakages, and run the test suites.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/upgrade-llvm-mlir.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Upgrade LLVM/MLIR (StableHLO-pinned)

Guided walkthrough for upgrading onnx-mlir's LLVM/MLIR dependency and StableHLO submodule to the commit that StableHLO currently tracks. Covers SHA discovery, change assessment, pin updates, API breakage fixing, rebuild, and testing.

---

## Overview

onnx-mlir pins LLVM at a specific commit (tracked in `utils/clone-mlir.sh`) and StableHLO as a submodule (`third_party/stablehlo`). The community updates both biweekly, using the SHA that StableHLO tracks from XLA as the canonical safe point.

This skill walks through the full upgrade in 8 phases. **Phase 1 ends with a scale assessment — if the upgrade is large, pause and ask the user whether to proceed.**

Files that carry the LLVM SHA and must be kept in sync:
- `utils/clone-mlir.sh` — primary pin
- `docs/BuildOnLinuxOSX.md` — mirrors the SHA in a code block (tagged `[same-as-file]`)
- `docs/BuildOnWindows.md` — same

---

## Phase 1: Discover Target SHAs and Assess Scale

### 1a. Find the target LLVM SHA

StableHLO publishes the LLVM SHA it tracks here:
```
https://raw.githubusercontent.com/openxla/stablehlo/main/build_tools/llvm_version.txt
```

```bash
curl -s https://raw.githubusercontent.com/openxla/stablehlo/main/build_tools/llvm_version.txt
# Example output: 87d42c13cd6b119240781f31e5869981d500a186
NEW_LLVM_SHA=<paste output here>
```

### 1b. Find the matching StableHLO commit

The StableHLO commit that goes with `$NEW_LLVM_SHA` is found by looking at the git log of `llvm_version.txt` in the StableHLO repo:

```bash
# In a local stablehlo checkout (or via gh api):
git -C third_party/stablehlo log --oneline --format="%H %s" -- build_tools/llvm_version.txt | head -10
# Find the commit that introduced $NEW_LLVM_SHA, then use that commit's parent's SHA as the StableHLO pin
```

Or look at recent StableHLO PRs titled "Integrate LLVM at <sha>":
```
https://github.com/openxla/stablehlo/commits/main/build_tools/llvm_version.txt
```

The StableHLO commit SHA that updated `llvm_version.txt` to `$NEW_LLVM_SHA` is your `$NEW_STABLEHLO_SHA`.

### 1c. Read the current pins

```bash
# Current LLVM SHA
CURRENT_LLVM_SHA=$(grep -oE '[0-9a-f]{40}' utils/clone-mlir.sh)   # line is "cd llvm-project && git checkout <sha> && cd .."
echo "Current LLVM: $CURRENT_LLVM_SHA"

# Current StableHLO SHA
CURRENT_STABLEHLO_SHA=$(git submodule status third_party/stablehlo | awk '{print $1}' | tr -d '-+')
echo "Current StableHLO: $CURRENT_STABLEHLO_SHA"
```

### 1d. Assess the scale of the upgrade

```bash
# Count MLIR-relevant commits between the two SHAs
# (run from inside the llvm-project checkout, after fetching)
cd /path/to/llvm-project
git fetch origin
MLIR_COMMIT_COUNT=$(git log --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/ | wc -l)
echo "MLIR commits in range: $MLIR_COMMIT_COUNT"

# Write the log for later reference
git log --reverse $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/ > /tmp/mlir-upgrade.log
echo "Full log written to /tmp/mlir-upgrade.log"
wc -l /tmp/mlir-upgrade.log
```

**→ USER FEEDBACK GATE**

Before continuing, report the following to the user and ask whether to proceed:

```
LLVM upgrade summary:
  Current SHA:   $CURRENT_LLVM_SHA (short: ${CURRENT_LLVM_SHA:0:8})
  Target SHA:    $NEW_LLVM_SHA     (short: ${NEW_LLVM_SHA:0:8})
  MLIR commits:  $MLIR_COMMIT_COUNT

Scale assessment:
  < 50  commits → routine update, proceed
  50–200 commits → moderate; scan for dialect API changes before proceeding
  > 200 commits → large upgrade; recommend scanning the log and filing a WIP PR if breakages are complex

Do you want to proceed with this upgrade?
```

Also scan the log for high-impact patterns and surface them:
```bash
# Ops renamed or removed (high breakage risk)
grep -iE "rename|remove|deprecat|break" /tmp/mlir-upgrade.log | grep -iE "affine|arith|memref|func|scf|math|vector|linalg|llvm|tensor" | head -20

# Type system changes (very high breakage risk)
grep -iE "type|TypeConverter|mlir::cast|dyn_cast|isa<" /tmp/mlir-upgrade.log | head -10

# Pass infrastructure changes
grep -iE "PassManager|OpPassManager|pipeline|addPass" /tmp/mlir-upgrade.log | head -10
```

If any of these grep results are non-empty, summarize them for the user before proceeding.

The local `git log` needs full history between the two SHAs. Without a full clone (e.g. a shallow or blobless checkout), get the total from the GitHub compare API — `total_commits` is exact, but the `commits` list it returns is capped at 250, so use it for the count only:
```bash
curl -s https://api.github.com/repos/llvm/llvm-project/compare/$CURRENT_LLVM_SHA...$NEW_LLVM_SHA \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["total_commits"])'
```
That count is all of llvm-project, not just `mlir/`. For scale: the `1053047a` → `43574226` bump (onnx-mlir `c41c212`) spanned 23,864 commits (2026-02-25 → 2026-08-27, LLVM 23 → 24 development) and needed five onnx-mlir source fixes plus CHECK-line regeneration in 65 lit tests.

---

## Phase 2: Scan MLIR Changes Relevant to onnx-mlir

Before touching any code, understand what changed. This phase informs Phase 5 (fixing breakages).

### 2a. Focus the log on dialects onnx-mlir uses

```bash
# onnx-mlir's primary MLIR surface: affine, arith, memref, func, scf, math, vector, linalg, llvm
git -C /path/to/llvm-project log --reverse --oneline \
  $CURRENT_LLVM_SHA..$NEW_LLVM_SHA \
  -- mlir/include/mlir/Dialect/Affine/ \
     mlir/include/mlir/Dialect/Arith/ \
     mlir/include/mlir/Dialect/MemRef/ \
     mlir/include/mlir/Dialect/Func/ \
     mlir/include/mlir/Dialect/SCF/ \
     mlir/include/mlir/Dialect/Math/ \
     mlir/include/mlir/Dialect/Vector/ \
     mlir/include/mlir/Dialect/Linalg/ \
     mlir/include/mlir/Dialect/LLVM/ \
     mlir/include/mlir/IR/ \
     mlir/include/mlir/Pass/ \
     mlir/include/mlir/Transforms/ \
  > /tmp/mlir-core-changes.log

echo "Core dialect/IR changes: $(wc -l < /tmp/mlir-core-changes.log)"
```

### 2b. Find specific API changes that affect onnx-mlir

Use `git log -G <pattern>` to find commits that modified a specific string. This is the most efficient technique for tracking down individual API breakages when the compiler reports them.

Key APIs onnx-mlir uses — search for changes to each if you encounter errors:

```bash
# ConversionPattern / TypeConverter (affects all dialect conversions)
git -C /path/to/llvm-project log -G "ConversionPattern\|TypeConverter\|matchAndRewrite" \
  --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/ | head -10

# OpBuilder / ImplicitLocOpBuilder
git -C /path/to/llvm-project log -G "OpBuilder\|ImplicitLocOpBuilder\|create<" \
  --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/include/ | head -10

# memref::AllocOp / DeallocOp (MemRefBuilder)
git -C /path/to/llvm-project log -G "AllocOp\|DeallocOp\|AllocaOp" \
  --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/include/mlir/Dialect/MemRef/ | head -10

# affine.for / affine.if
git -C /path/to/llvm-project log -G "AffineForOp\|AffineIfOp\|affine::For" \
  --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/include/mlir/Dialect/Affine/ | head -10

# PassManager pipeline API
git -C /path/to/llvm-project log -G "OpPassManager\|addNestedPass\|addPass\b" \
  --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/include/mlir/Pass/ | head -10

# LowerToLLVM / ConvertToLLVM pass names
git -C /path/to/llvm-project log -G "LowerToLLVM\|ConvertToLLVM\|FinalizeMemRef" \
  --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/ | head -10
```

For any commit ID that looks relevant, view the full diff:
```bash
git -C /path/to/llvm-project show <COMMIT_SHA> -- mlir/ | head -100
```

### 2c. Check MLIR migration docs

MLIR sometimes publishes deprecation notices. Check:
```bash
git -C /path/to/llvm-project log --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA \
  -- mlir/docs/ | head -20
# If there are deprecation or migration docs, read them:
git -C /path/to/llvm-project show <doc-commit> -- mlir/docs/
```

---

## Phase 3: Update the Pin Files

```bash
# In onnx-mlir repo root:
OLD_SHA=$CURRENT_LLVM_SHA
NEW_SHA=$NEW_LLVM_SHA

# 1. update utils/clone-mlir.sh
sed -i "s/$OLD_SHA/$NEW_SHA/g" utils/clone-mlir.sh
grep "git checkout" utils/clone-mlir.sh  # verify

# 2. update docs/BuildOnLinuxOSX.md (tagged [same-as-file] — must match clone-mlir.sh)
sed -i "s/$OLD_SHA/$NEW_SHA/g" docs/BuildOnLinuxOSX.md
grep "$NEW_SHA" docs/BuildOnLinuxOSX.md  # verify

# 3. update docs/BuildOnWindows.md
sed -i "s/$OLD_SHA/$NEW_SHA/g" docs/BuildOnWindows.md
grep "$NEW_SHA" docs/BuildOnWindows.md  # verify

# 4. update third_party/stablehlo submodule
cd third_party/stablehlo
git fetch origin
git checkout $NEW_STABLEHLO_SHA
cd ../..
git add third_party/stablehlo
```

Verify all three SHA occurrences are updated:
```bash
grep -r "$OLD_SHA" utils/ docs/ third_party/stablehlo/ 2>/dev/null
# should print nothing — old SHA is gone
```

---

## Phase 4: Rebuild LLVM at the New SHA

**This is the slow step (~1 hour for a full build). If the LLVM directory is the same checkout, just fetch and re-checkout.**

```bash
cd /path/to/llvm-project
git fetch origin
git checkout $NEW_LLVM_SHA

# Full rebuild (required if you haven't built this SHA before)
cd build
cmake --build . -- ${MAKEFLAGS}

# Optionally run MLIR tests to verify the new LLVM is healthy
cmake --build . --target check-mlir 2>&1 | tail -20
```

**Note on CI**: Jenkins may time out on the first build due to cache rebuilding. If this happens during CI, restart the build — it will use the cached LLVM. Contact `@gongsu832` or `@AlexandreEichenberger` if Jenkins disk space is the issue.

---

## Phase 5: Fix API Breakages in onnx-mlir

Rebuild onnx-mlir against the new LLVM. The first build will produce a list of compiler errors. Address them systematically.

```bash
cd /path/to/onnx-mlir/build
cmake .. -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build . 2>&1 | tee /tmp/build-errors.log

# Count and classify errors
grep "error:" /tmp/build-errors.log | sed "s|.*onnx-mlir/||" | cut -d: -f1 | sort | uniq -c | sort -rn | head -20
```

### 5a. Error classification

Group errors by source file:
```bash
grep "error:" /tmp/build-errors.log | grep -oP "src/[^:]+\.cpp" | sort | uniq -c | sort -rn | head -20
```

Files most frequently affected by MLIR API changes in onnx-mlir:

| File area | Common breakage types |
|-----------|----------------------|
| `src/Dialect/Mlir/DialectBuilder.hpp/.cpp` | `OpBuilder` method signature changes, `create<>` template changes |
| `src/Conversion/KrnlToAffine/` | `AffineForOp`/`AffineIfOp` API, affine map construction |
| `src/Conversion/KrnlToLLVM/` | `LLVMTypeConverter`, `ConvertToLLVMPattern`, pass registration |
| `src/Compiler/CompilerPasses.cpp` | Pass names renamed, `OpPassManager::addPass` signature |
| `src/Dialect/ONNX/Transforms/` | `PatternRewriter`, `RewritePattern` registration |
| `src/Conversion/ONNXToKrnl/` | `ConversionPattern`, `TypeConverter`, `matchAndRewrite` |
| `src/Dialect/Krnl/KrnlOps.cpp` | Op definition interface changes |
| `src/Conversion/KrnlToLLVM/KrnlInstrument.cpp` | `LLVM::DI*Attr` builder signatures (debug-info attributes) |
| `src/Compiler/CompilerOptions.cpp` | `llvm::cl` registered-option map iteration (ADT iterator-invalidation assertions) |
| `test/mlir/lit.cfg.py` | lit API (`lit.formats.ShTest` arguments) |

### 5b. Fix patterns — follow MLIR's own migration style

When fixing an API breakage, **search how MLIR itself migrated** to the new API. This ensures onnx-mlir adopts the canonical new pattern rather than a workaround.

```bash
# For any renamed/changed API, find the MLIR commit that did the migration:
git -C /path/to/llvm-project log -G "<old-api-name>" \
  --oneline $CURRENT_LLVM_SHA..$NEW_LLVM_SHA -- mlir/ | head -5

# Read the diff to see exactly how the migration was done:
git -C /path/to/llvm-project show <migration-commit-sha> -- mlir/ | \
  grep -A5 -B5 "<old-api-name>" | head -40
```

The pattern from the migration commit should be applied identically in onnx-mlir.

#### Common migration patterns (verify against actual diff)

**`OpBuilder::create<Op>` argument changes:**
```cpp
// Old: builder.create<memref::AllocOp>(loc, type, dynSizes)
// New: check the MLIR diff — builder signature may require named args or changed order
```

**Pass name changes (check `CompilerPasses.cpp`):**
```cpp
// Old: pm.addPass(mlir::createLowerToLLVMPass())
// New: check git log -G "LowerToLLVM\|createConvertToLLVM" -- mlir/ for the rename commit
```

**`isa<>`/`dyn_cast<>`/`cast<>` → `mlir::isa<>`/`mlir::dyn_cast<>`/`mlir::cast<>` migration:**
This is a recurring pattern in LLVM upgrades — using the qualified form.

#### Adaptations made in the `43574226` upgrade (onnx-mlir `c41c212`)

Concrete instances of the patterns above, useful as a checklist of failure *shapes* — several show up as verifier errors or assertions at test time, not compile errors:

| Symptom | Cause upstream | Fix in onnx-mlir |
|---------|----------------|------------------|
| No matching `DICompileUnitAttr::get` overload | 9-arg builder removed; attribute gained `recId`/`isRecSelf`, `DISourceLanguageNameAttr`, `isDebugInfoForProfiling`, `importedEntities` | `KrnlInstrument.cpp` calls the full builder, wrapping `DW_LANG_C99` in `DISourceLanguageNameAttr::get(ctx, lang, 0, std::nullopt, 0)` |
| Assertion while iterating a map in an assertions-enabled build | ADT map iterators are epoch-checked; erasing during range-for asserts | `removeUnrelatedOptions()` collects options first, then calls `removeArgument()` |
| `llvm.call` verifier reports an argument-count mismatch on variadic runtime calls | `llvm.call` derives fixed arguments from `var_callee_type` | `LLVMBuilder::handleVarArgCall` uses the callee `llvm.func`'s declared type |
| `type of return operand ... doesn't match function result type` after `--constprop-onnx` | `func.func` region verifier now checks `onnx.Return` too | `ConstPropONNXToONNXPass` updates the function type after the greedy rewrite |
| `memref.subview` fails to legalize in `ConvertKrnlToLLVM` ("subview operations should have been expanded beforehand") | MemRefToLLVM requires expanded subviews | add `memref::populateExpandStridedMetadataPatterns()` to the pattern set |
| lit raises `ValueError` before running any test | `ShTest(execute_external=True)` no longer accepted | `lit.formats.ShTest()` |

**ConversionPattern `matchAndRewrite` signature:**
```cpp
// Occasionally the last argument changes between ConversionPatternRewriter& and PatternRewriter&
// Check: git log -G "matchAndRewrite" -- mlir/include/mlir/Transforms/DialectConversion.h
```

### 5c. Iterate build → fix → build

Repeat until clean:
```bash
cmake --build . 2>&1 | grep "error:" | head -30
# Fix each error, then rebuild
```

**→ USER FEEDBACK GATE (if applicable)**

If after 2–3 iterations there are still >10 errors, or if errors are in complex areas (NNPA/ZHigh conversions, pass infrastructure), pause and report to the user:

```
Build status after fixes:
  Remaining errors: N
  Affected files: [list]
  
Most complex remaining issues: [describe]

Recommendation: file a WIP PR and ask for help from the community,
OR continue if the fixes are mechanical (see docs/UpdatingLLVMCommit.md).
```

---

## Phase 7: Run the Test Suite

Run tests in order of speed. Stop and investigate on first failure rather than running all tests blindly.

### 7a. Lit tests (fast, ~minutes)
```bash
cmake --build . --target check-onnx-lit 2>&1 | tail -30
```
These are FileCheck tests on MLIR output. An LLVM upgrade commonly breaks these if pass output format changed (e.g., op names, attribute syntax). In the `43574226` upgrade the changes were all output-format or canonicalization changes: `memref.alloc() alignment = 16` instead of `{alignment = 16 : i64}`, `arith.cmpi`+`arith.select` → `arith.maxsi`/`minsi`, `sitofp(extsi(x))` → `sitofp(x)`, and `vector.from_elements` shapes.

When regenerating CHECK lines for a file with several functions, check for FileCheck variable-name collisions across functions (`c41c212` had to fix these in NNPA lower-to-llvm tests). Lit tests added on `main` after the bulk regeneration but before the upgrade merges still use the old syntax — re-run `check-onnx-lit` after rebasing.

**If lit tests fail:** Compare the IR output to the expected `.mlir` FileCheck patterns in `test/`. Update the patterns to match the new op format. Use:
```bash
cmake --build . --target check-onnx-lit 2>&1 | grep "FAILED\|error:" | head -20
# Then: open the failing .mlir file and update the // CHECK: lines
```

### 7b. Numerical tests (medium, ~10 minutes)
```bash
cmake --build . --target check-onnx-numerical 2>&1 | tail -30
```

### 7c. Backend tests: static shapes (the CI gate)
```bash
cmake --build . --target check-onnx-backend 2>&1 | tail -50
```

### 7d. Backend tests: dynamic shapes
```bash
cmake --build . --target check-onnx-backend-dynamic 2>&1 | tail -30
```

### 7e. Backend tests: constant inputs
```bash
cmake --build . --target check-onnx-backend-constant 2>&1 | tail -30
```

### 7f. Single op test (when debugging a specific failure)
```bash
TEST_CASE_BY_USER=test_add_cpu cmake --build . --target check-onnx-backend
```

### 7g. CI-equivalent full suite

The upstream CI runs these targets (from `.circleci/config.yml` and `.github/workflows/`):
- `check-onnx-lit`
- `check-onnx-numerical`
- `check-onnx-backend`
- `check-onnx-backend-dynamic`
- `check-onnx-backend-constant`
- `check-onnx-backend-signature`

`c41c212` also reported `check-onnx-backend-node` alongside these.

On s390x, also:
- `TEST_MARCH=z16 cmake --build . --target check-onnx-backend` (NNPA)

**→ USER FEEDBACK GATE**

After running tests, report the summary:
```
Test results:
  check-onnx-lit:            PASS / FAIL (N failures)
  check-onnx-numerical:      PASS / FAIL
  check-onnx-backend:        PASS / FAIL (N failures)
  check-onnx-backend-dynamic: PASS / FAIL

Are failures blocking (regression) or pre-existing?
Recommend: proceed to PR only if check-onnx-lit and check-onnx-backend pass.
```

---

## Phase 8: Prepare the PR

### 8a. Minimal commit scope

The LLVM update PR should contain **only**:
1. `utils/clone-mlir.sh` — LLVM SHA update
2. `docs/BuildOnLinuxOSX.md` — SHA update
3. `docs/BuildOnWindows.md` — SHA update
4. `third_party/stablehlo` — submodule update
5. Any onnx-mlir source fixes required by the API changes
6. Any `.mlir` FileCheck pattern updates

Do **not** bundle unrelated fixes, new features, or refactors.

### 8b. Commit message convention
```
[LLVM] Update LLVM/MLIR to <new-sha-short> and StableHLO to <stablehlo-sha-short>

- llvm-project: <current-sha-short> → <new-sha-short>
- third_party/stablehlo: <current-stablehlo-short> → <new-stablehlo-short>

API changes addressed:
- <describe each non-trivial fix, referencing the MLIR commit that introduced it>

Fixes: <link to any issues if applicable>
```

### 8c. PR description checklist
- [ ] LLVM SHA updated in all 3 locations (clone-mlir.sh, BuildOnLinuxOSX.md, BuildOnWindows.md)
- [ ] StableHLO submodule updated
- [ ] check-onnx-lit passes
- [ ] check-onnx-backend passes
- [ ] PR title: `[LLVM] Update LLVM to <short-sha> and StableHLO to <short-sha>`

Recent example: onnx-mlir `c41c212`, titled "Upgrade LLVM to 43574226b712 and StableHLO to 63479dd8 (#3658)", whose message lists each API breakage fixed, the lit-infrastructure changes, and the pass count per test target. Older example: https://github.com/onnx/onnx-mlir/pull/2662

---

## Quick Reference: Key URLs

| Resource | URL |
|----------|-----|
| StableHLO LLVM pin | https://raw.githubusercontent.com/openxla/stablehlo/main/build_tools/llvm_version.txt |
| StableHLO llvm_version history | https://github.com/openxla/stablehlo/commits/main/build_tools/llvm_version.txt |
| Jenkins LLVM watch (s390x) | https://www.onnxmlir.xyz/jenkins/job/LLVM-Watch-Docker-Build/ |
| Jenkins LLVM watch (amd64) | https://www.onnxmlir.xyz/jenkinx/job/LLVM-Watch-Docker-Build/ |
| Update rotation | https://github.com/onnx/onnx-mlir/wiki/LLVM-and-StableHLO-Update-Schedule |
| Example upgrade PRs | https://github.com/onnx/onnx-mlir/pull/3658, https://github.com/onnx/onnx-mlir/pull/2662 |

---

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:docs/UpdatingLLVMCommit.md`, `repo:onnx-mlir@b505d64:utils/clone-mlir.sh`, `repo:onnx-mlir@b505d64:docs/BuildOnLinuxOSX.md`, `repo:onnx-mlir@b505d64:docs/BuildOnWindows.md`, `repo:onnx-mlir@b505d64:docs/Testing.md`, onnx-mlir commit `c41c212` (diff and message), `repo:llvm-project@43574226:mlir/` (API changes in the adaptation table)
- **Files read:** `utils/clone-mlir.sh` (SHA at `43574226`), `docs/BuildOnLinuxOSX.md` line 18 and `docs/BuildOnWindows.md` line 55 (same SHA), `docs/UpdatingLLVMCommit.md` (full update process), `docs/Testing.md` (test targets), `.github/workflows/`, `.circleci/config.yml` (CI targets); for the adaptation table, `src/Conversion/KrnlToLLVM/{KrnlInstrument,ConvertKrnlToLLVM}.cpp`, `src/Compiler/CompilerOptions.cpp`, `src/Dialect/Mlir/DialectBuilder.cpp`, `src/Dialect/ONNX/Transforms/ConstProp.cpp`, `test/mlir/lit.cfg.py` at `5534bc7`, and their upstream `llvm-project` counterparts
- **Re-checked at `repo:onnx-mlir@b505d64`:** none of the files listed above changed between `5534bc7` and `b505d64` except `src/Compiler/CompilerOptions.cpp`, which gained two options (`--kv-cache`, `--disable-unbind-to-split`) without touching the option-map iteration in the adaptation table; the LLVM pin is still `43574226`. `.buildbot/Jenkinsfile` changed only its lock-file directory.
- **How to test:** Execute Phase 1 to verify SHA discovery works; Phase 7 test targets are the gate
- **Staleness risk:** `utils/clone-mlir.sh` SHA changes with every update — re-read it before running this skill. The test targets are stable (cmake targets defined in CMakeLists.txt).
