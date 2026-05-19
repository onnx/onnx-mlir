#!/usr/bin/env python3
"""
Script to fix // ----- separators in .mlir files.
Ensures there's always an empty line before and after each separator.
Also adds missing separators between consecutive functions.

WARNING: This script makes automated changes that may not handle all corner cases.
         ALL CHANGES MUST BE MANUALLY REVIEWED before committing!

Run this command in the onnx-mlir/test/mlir directory,
e.g. python ../../utils/fix_mlir_separators.py
"""

import sys
import re
from pathlib import Path

# Track all modified files
modified_files = []


def fix_separator_spacing(content):
    """Fix spacing around // ----- separators and add missing separators between functions."""
    lines = content.split("\n")
    fixed_lines = []
    i = 0
    last_run_line = -1

    while i < len(lines):
        line = lines[i]

        # Track RUN commands
        if line.strip().startswith("// RUN:"):
            last_run_line = len(fixed_lines)

        # Check if current line is a separator
        if line.strip() == "// -----":
            # Ensure exactly one empty line before separator
            # Remove multiple consecutive empty lines before separator
            while (
                fixed_lines
                and fixed_lines[-1].strip() == ""
                and len(fixed_lines) > 1
                and fixed_lines[-2].strip() == ""
            ):
                fixed_lines.pop()

            # Add empty line before if needed
            if fixed_lines and fixed_lines[-1].strip() != "":
                fixed_lines.append("")

            # Add the separator
            fixed_lines.append(line)

            # Check if we need empty line after
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                fixed_lines.append("")
        else:
            fixed_lines.append(line)

            # Check if we need to add a separator after RUN commands
            # Look for the end of RUN command block (non-RUN, non-empty line after RUN)
            if (
                last_run_line >= 0
                and line.strip()
                and not line.strip().startswith("// RUN:")
            ):
                # Check if there's already a separator
                has_separator = False
                for j in range(last_run_line, len(fixed_lines)):
                    if fixed_lines[j].strip() == "// -----":
                        has_separator = True
                        break

                if not has_separator and not line.strip().startswith("//"):
                    # Remove multiple empty lines after RUN block
                    insert_pos = len(fixed_lines) - 1
                    while insert_pos > 0 and fixed_lines[insert_pos - 1].strip() == "":
                        fixed_lines.pop(insert_pos - 1)
                        insert_pos -= 1

                    # Insert separator after RUN block with proper spacing
                    insert_pos = len(fixed_lines) - 1
                    fixed_lines.insert(insert_pos, "")
                    fixed_lines.insert(insert_pos + 1, "// -----")
                    fixed_lines.insert(insert_pos + 2, "")

                last_run_line = -1  # Reset

            # Check if we need to add a separator between functions
            # Look for closing brace followed by func.func (with possible empty lines/comments in between)
            if line.strip() == "}" and i + 1 < len(lines):
                # Look ahead to see if there's a func.func coming
                j = i + 1
                found_func = False
                found_separator = False
                empty_or_comment_only = True

                while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                    next_line = lines[j].strip()
                    if next_line == "// -----":
                        found_separator = True
                        break
                    elif next_line.startswith("func.func"):
                        found_func = True
                        break
                    elif next_line and not next_line.startswith("//"):
                        # Found non-comment, non-empty content that's not a function
                        empty_or_comment_only = False
                        break
                    j += 1

                # If we found a function without a separator, add one
                if found_func and not found_separator and empty_or_comment_only:
                    # Add empty line if the last line wasn't empty
                    if fixed_lines and fixed_lines[-1].strip() != "":
                        fixed_lines.append("")
                    fixed_lines.append("// -----")
                    fixed_lines.append("")

        i += 1

    return "\n".join(fixed_lines)


def process_file(filepath):
    """Process a single .mlir file."""
    global modified_files
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        fixed_content = fix_separator_spacing(content)

        # Only write if content changed
        if fixed_content != content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            print(f"Fixed: {filepath}")
            modified_files.append(str(filepath))
            return True

        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    global modified_files

    print("=" * 80)
    print("WARNING: This script makes automated changes to .mlir files.")
    print("         Not all corner cases are handled correctly.")
    print("         ALL CHANGES MUST BE MANUALLY REVIEWED before committing!")
    print("=" * 80)
    print()

    if len(sys.argv) > 1:
        # Process specific files
        files = [Path(f) for f in sys.argv[1:]]
    else:
        # Find all .mlir files recursively
        files = Path(".").rglob("*.mlir")

    fixed_count = 0
    for filepath in files:
        if process_file(filepath):
            fixed_count += 1

    print()
    print("=" * 80)
    print(f"Total files modified: {fixed_count}")
    print("=" * 80)

    if modified_files:
        print("\nModified files:")
        for f in modified_files:
            print(f"  - {f}")
        print()
        print("=" * 80)
        print("IMPORTANT: Review all changes with 'git diff' before committing!")
        print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
