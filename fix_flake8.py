#!/usr/bin/env python3
"""
Script to fix common Flake8 issues automatically.
Fixes: trailing whitespace, blank lines with whitespace, and some basic formatting.
"""

import os


def fix_file(filepath):
    """Fix common Flake8 issues in a single file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Fix W291: trailing whitespace
        line = line.rstrip() + '\n'
        
        # Fix W293: blank line contains whitespace (keep only \n for blank lines)
        if line.strip() == '':
            line = '\n'
        
        fixed_lines.append(line)
    
    # Remove W391: blank line at end of file
    while fixed_lines and fixed_lines[-1].strip() == '':
        fixed_lines.pop()
    
    # Add single newline at end of file if content exists
    if fixed_lines and not fixed_lines[-1].endswith('\n'):
        fixed_lines[-1] = fixed_lines[-1] + '\n'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed: {filepath}")


def main():
    """Fix Flake8 issues in all Python files."""
    # Get all Python files in src/ and tests/
    python_files = []
    
    for root, _, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    for root, _, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files to fix")
    
    for filepath in python_files:
        fix_file(filepath)
    
    print("Done! Run flake8 again to check remaining issues.")


if __name__ == "__main__":
    main()
