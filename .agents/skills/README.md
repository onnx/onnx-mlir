# Agent skills for onnx-mlir

Task guides (`SKILL.md` files) for AI coding agents working in this
repository: adding ops and lowerings, writing passes and lit tests,
debugging compilation, NNPA performance, and more. Agents that support the
Agent Skills format load a skill when its `description` matches the task.
`onnx-mlir-internals` and `nnpa-internals` hold background reference pages
that the task skills link to.

## Where these come from

These files are generated from the onnx-mlir knowledge base (OMKB),
maintained by the IBM onnx-mlir team, where each skill is grounded against
the code at a pinned commit; the banner at the top of each file names it.
They are regenerated after the knowledge base syncs to a newer onnx-mlir
commit, so they can trail `main` slightly.

**Please don't edit them here.** Hand edits are overwritten by the next
export. If a skill is wrong or missing something, open an issue or PR that
describes the fix; it will be folded into the knowledge base and come back
through the next export.

## Agents that read another directory

Some agents look for skills in their own directory rather than
`.agents/skills/`. The repository commits a symlink for each, so they
find the same files with no setup:

| Agent       | Reads skills from | Committed link        |
|-------------|-------------------|-----------------------|
| Claude Code | `.claude/skills/` | `-> ../.agents/skills` |
| IBM Bob     | `.bob/skills/`    | `-> ../.agents/skills` |

On Windows, git checks a symlink out as a plain text file unless
`core.symlinks` is enabled; there, recreate the link by hand or point the
agent at `.agents/skills/`.
