#!/usr/bin/env python3
"""
Script to fix // ----- separators in .mlir files.
Ensures there's always an empty line before and after each separator.
Run this command in the onnx-mlir/test/mlir directory,
e.g. python ../../utils/fix_mlir_separators.py
"""

import sys
import re
from pathlib import Path


def fix_separator_spacing(content):
    """Fix spacing around // ----- separators and add missing separators between functions."""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if current line is a separator
        if line.strip() == '// -----':
            # Check if we need empty line before
            if fixed_lines and fixed_lines[-1].strip() != '':
                fixed_lines.append('')
            
            # Add the separator
            fixed_lines.append(line)
            
            # Check if we need empty line after
            if i + 1 < len(lines) and lines[i + 1].strip() != '':
                fixed_lines.append('')
        else:
            fixed_lines.append(line)
            
            # Check if we need to add a separator between functions
            # Look for closing brace followed by func.func (with possible empty lines/comments in between)
            if line.strip() == '}' and i + 1 < len(lines):
                # Look ahead to see if there's a func.func coming
                j = i + 1
                found_func = False
                found_separator = False
                empty_or_comment_only = True
                
                while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                    next_line = lines[j].strip()
                    if next_line == '// -----':
                        found_separator = True
                        break
                    elif next_line.startswith('func.func'):
                        found_func = True
                        break
                    elif next_line and not next_line.startswith('//'):
                        # Found non-comment, non-empty content that's not a function
                        empty_or_comment_only = False
                        break
                    j += 1
                
                # If we found a function without a separator, add one
                if found_func and not found_separator and empty_or_comment_only:
                    # Add empty line if the last line wasn't empty
                    if fixed_lines and fixed_lines[-1].strip() != '':
                        fixed_lines.append('')
                    fixed_lines.append('// -----')
                    fixed_lines.append('')
        
        i += 1
    
    return '\n'.join(fixed_lines)


def process_file(filepath):
    """Process a single .mlir file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file contains separators
        if '// -----' not in content:
            return False
        
        fixed_content = fix_separator_spacing(content)
        
        # Only write if content changed
        if fixed_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed: {filepath}")
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) > 1:
        # Process specific files
        files = [Path(f) for f in sys.argv[1:]]
    else:
        # Find all .mlir files recursively
        files = Path('.').rglob('*.mlir')
    
    fixed_count = 0
    for filepath in files:
        if process_file(filepath):
            fixed_count += 1
    
    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == '__main__':
    main()

# Made with Bob
