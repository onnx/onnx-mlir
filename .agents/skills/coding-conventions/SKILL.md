---
name: coding-conventions
description: 'Use before finishing any new or edited onnx-mlir source file or commit: copyright header, clang-format/black formatting, comment style, and DCO sign-off, separating what CI enforces from convention.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/coding-conventions.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Coding Conventions Checklist

Recipe to run before finishing any new or edited onnx-mlir source file: copyright header, C++/Python formatting, comment style, and commit sign-off. Separates what CI actually machine-checks from what is only convention, since assuming the latter is enforced leads to silently-wrong code.

---

## What CI actually enforces

- **C++ formatting** — `.github/workflows/clang-format-check.yml` runs `clang-format` **v9** over `./src` only, using the repo's [`.clang-format`](https://github.com/onnx/onnx-mlir/blob/main/.clang-format):
  ```yaml
  BasedOnStyle: LLVM
  AlwaysBreakTemplateDeclarations: Yes
  AlignAfterOpenBracket: DontAlign
  ```
- **Python formatting** — `.github/workflows/black-format-check.yml` runs `black --check` over the whole repo except `third_party`.
- **Manual style rules** (reviewer-enforced, not machine-checked) — [`CODING_PRACTICE.md`](https://github.com/onnx/onnx-mlir/blob/main/CODING_PRACTICE.md) lists: use C++-style casts (`static_cast<int64_t>(1)`, not `(int64_t)1`); do bitwise ops on unsigned types; check the return of `malloc()`, `fopen()`, `fprintf()`; parenthesize macro parameter names.
- **Commit sign-off** — [`docs/Workflow.md`](https://github.com/onnx/onnx-mlir/blob/main/docs/Workflow.md) Step 7 requires DCO sign-off on every commit (`git commit -s`); this is unrelated to code formatting but gates every PR the same way.

## What is *not* enforced anywhere in this repo

Verified by reading `utils/` in full (no header/license-update script exists there) and `CODING_PRACTICE.md`/`CONTRIBUTING.md` (neither mentions comment prose or copyright dates):

- **No script bumps the copyright year.** The header convention, confirmed empirically in `src/Dialect/ONNX/ONNXOps.cpp`, is:
  ```cpp
  /*
   * SPDX-License-Identifier: Apache-2.0
   */

  //===------------------ <File>.cpp - <One-line description> ---------------------===//
  //
  // Copyright <start-year>-<last-edit-year> The IBM Research Authors.
  //
  // =============================================================================
  //
  // <Longer description of what this file provides.>
  //
  //===----------------------------------------------------------------------===//
  ```
  Nothing in CI or `utils/` extends `<last-edit-year>` automatically — files touched today can still show an old end-year unless someone edits it by hand. Do not assume this is handled.
- **No linter checks comment prose.** "Comments are complete sentences: capitalized, ending in a period" is the general LLVM Coding Standards convention (onnx-mlir's `.clang-format` is `BasedOnStyle: LLVM`), but it is not machine-checked in this repo — clang-format reformats whitespace/wrapping only, it does not touch comment text content. Treat this as a self-check, not something a tool will catch in review.

## Steps to apply when writing/editing a file

1. **New file** — prepend the SPDX + copyright banner shown above, with this year as both start and end year if it's a brand-new file.
2. **Existing file** — look at the current `Copyright <start>-<end>` line. If `<end>` is not the current year, extend it (e.g. `2019-2023` → `2019-2026`) by hand; nothing does this for you.
3. **C++ changes** — run `clang-format -i <file>` before committing (matches the CI check, which pins clang-format v9 against `./src`).
4. **Python changes** — run `black <file>` before committing (matches the CI check, repo-wide except `third_party`).
5. **Every new/edited comment** — self-check it starts with a capital letter and ends with a period; this is a manual pass, no tool flags it here.
6. **Commit** — use `git commit -s` (DCO requirement, checked on every PR).

None of steps 2 or 5 are automatable from *this* repo's own tooling — if that matters, it needs a local pre-commit hook or editor-integration script outside onnx-mlir itself.

---

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:.clang-format`, `CODING_PRACTICE.md`, `CONTRIBUTING.md`, `docs/Workflow.md`, `.github/workflows/clang-format-check.yml`, `.github/workflows/black-format-check.yml`, `utils/` (full directory listing), `src/Dialect/ONNX/ONNXOps.cpp` (header example)
- **Files read:** all of the above, fetched at the pinned sync SHA `cf0842f3ebb13c0b151ee84b2c2816e5f6e1c3d5`
- **Re-checked at:** `repo:onnx-mlir@b505d64` — the same files: `.clang-format` (three keys unchanged), `clang-format-check.yml` (`source: ./src`, `clangFormatVersion: 9`), `black-format-check.yml` (`--check --exclude third_party`), `CODING_PRACTICE.md` (the four manual rules), `docs/Workflow.md` ("Step 7: Commit & Push", `git commit -s`), the `utils/` listing (still no copyright-year updater; files there only carry the banner), and `src/Dialect/ONNX/ONNXOps.cpp`'s header (`Copyright 2019-2023`, an example of an end-year nobody bumped)
- **How to test:** `clang-format --dry-run -Werror -style=file <changed.cpp>` should report no diff against the repo's `.clang-format`; `black --check <changed.py>` should pass; grep the file's copyright line and confirm the end-year matches today's year.
