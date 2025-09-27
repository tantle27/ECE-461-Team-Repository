#!/usr/bin/env python
import subprocess
import sys
import re

def main():
    # Run pytest with coverage
    result = subprocess.run([
        sys.executable, '-m', 'pytest', '--cov=src', '--cov-report=term', 'tests/'
    ], capture_output=True, text=True)
    output = result.stdout + '\n' + result.stderr

    # Parse test results
    test_match = re.search(r'(\d+) passed.*?in [\d.]+s', output, re.DOTALL)
    fail_match = re.search(r'(\d+) failed', output)
    skip_match = re.search(r'(\d+) skipped', output)
    total_tests = 0
    passed = 0
    if test_match:
        passed = int(test_match.group(1))
        total_tests = passed
        if fail_match:
            total_tests += int(fail_match.group(1))
        if skip_match:
            total_tests += int(skip_match.group(1))
    else:
        print("Could not parse test results.")
        print(output)
        sys.exit(2)

    # Parse coverage percentage from the summary table
    cov_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
    coverage = int(cov_match.group(1)) if cov_match else 0

    # Print required output line always
    print(f"{passed}/{total_tests} test cases passed. {coverage}% line coverage achieved.")

    # Check requirements
    success = True
    if total_tests < 20:
        print(f"Test suite has only {total_tests} test cases (minimum 20 required).", file=sys.stderr)
        success = False
    if coverage < 80:
        print(f"Line coverage is only {coverage}% (minimum 80% required).", file=sys.stderr)
        success = False
    if not success:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
