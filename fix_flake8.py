#!/usr/bin/env python3
"""Script to fix Flake8 issues automatically."""

import re
import os

def fix_long_lines(file_path):
    """Fix long lines in the given file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix specific long lines in handlers.py
    if 'handlers.py' in file_path:
        # Fix the rate limit messages
        content = re.sub(
            r'ctx\.fetch_logs\.append\(f"Rate limited, retrying in \{retry_delay\}s\.\.\."\)',
            'msg = f"Rate limited, retrying in {retry_delay}s..."\n                    ctx.fetch_logs.append(msg)',
            content
        )
        
        # Fix GitHub API call
        content = re.sub(
            r'response = requests\.get\(f"https://api\.github\.com/repos/\{owner\}/\{repo\}"\)',
            'url = f"https://api.github.com/repos/{owner}/{repo}"\n                response = requests.get(url)',
            content
        )
        
        # Fix GitHub API error message
        content = re.sub(
            r'raise Exception\(f"GitHub API error: \{response\.status_code\}"\)',
            'error_msg = f"GitHub API error: {response.status_code}"\n                    raise Exception(error_msg)',
            content
        )
    
    # Fix line length issues in error_handling.py
    if 'error_handling.py' in file_path:
        # Fix the comment line
        content = re.sub(
            r'Tests cover invalid URLs, API failures, missing files, and other error scenarios',
            'Tests cover invalid URLs, API failures, missing files,\nand other error scenarios',
            content
        )
        
        # Fix the if condition for checking file existence
        content = re.sub(
            r'if not file_path\.startswith\(\(\'http://\', \'https://\'\)\) or len\(file_path\) < 10:',
            'starts_with_http = file_path.startswith((\'http://\', \'https://\'))\n                if not starts_with_http or len(file_path) < 10:',
            content
        )
    
    # Fix integration.py long line
    if 'integration.py' in file_path:
        content = re.sub(
            r'metric_name = f"metric_\{i\}"',
            'metric_name = f"metric_{i}"',
            content
        )
        content = re.sub(
            r'metrics\.append\(SimpleTestMetric\(metric_name, weight=1\.0/num_metrics\)\)',
            'weight = 1.0/num_metrics\n            metrics.append(SimpleTestMetric(metric_name, weight=weight))',
            content
        )
    
    # Remove unused imports
    if 'error_handling.py' in file_path:
        content = re.sub(r'from unittest\.mock import patch, MagicMock\n', 'from unittest.mock import patch\n', content)
    
    # Fix import positions (move to top)
    if any(test_file in file_path for test_file in ['test_base_metric.py', 'test_integration.py', 'test_metric_eval.py']):
        lines = content.split('\n')
        import_lines = []
        other_lines = []
        found_first_import = False
        
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
                found_first_import = True
            elif found_first_import and line.strip() == '':
                import_lines.append(line)
            elif line.startswith('# Add src to path'):
                import_lines.append(line)
            elif line.startswith('sys.path.insert'):
                import_lines.append(line)
            else:
                other_lines.append(line)
        
        if import_lines and other_lines:
            # Reconstruct with imports at top
            content = '\n'.join(import_lines + [''] + other_lines)
    
    # Fix comment formatting issues
    content = re.sub(r'^(\s*)#{5,}(.*)$', r'\1# \2', content, flags=re.MULTILINE)
    
    # Write back if content changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Fixed issues in {file_path}')

def fix_unused_variables(file_path):
    """Fix unused variables."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix unused variables in error_handling.py
    if 'error_handling.py' in file_path:
        # Remove unused variable assignments
        content = re.sub(r'\s+with open\(missing_file_path, \'r\'\) as f:\s+pass', 
                        '                open(missing_file_path, \'r\')', content)
        content = re.sub(r'except FileNotFoundError as e:', 
                        'except FileNotFoundError:', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Fixed unused variables in {file_path}')

# Process all Python files
for root, dirs, files in os.walk('.'):
    if any(skip in root for skip in ['venv', '__pycache__', '.git']):
        continue
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            try:
                fix_long_lines(file_path)
                fix_unused_variables(file_path)
            except Exception as e:
                print(f'Error processing {file_path}: {e}')

print("Flake8 fixes completed!")
