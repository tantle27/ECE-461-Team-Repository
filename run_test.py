#!/usr/bin/env python3
"""
Test runner script for the ACME Corporation AI/ML Model Evaluation System.

This script provides a convenient way to run all tests w/ proper configuration.
"""

import sys
import subprocess
import os


def run_tests():
    """Run all tests using pytest."""
    print("🧪 Running ACME AI/ML Model Evaluation System Tests")
    print("=" * 60)

    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Run pytest with configuration
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n❌ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error running tests: {e}")
        return 1


def run_specific_test(test_file):
    """Run a specific test file."""
    print(f"🧪 Running specific test: {test_file}")
    print("=" * 60)

    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/{test_file}",
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        if not test_file.startswith("test_"):
            test_file = f"test_{test_file}"
        if not test_file.endswith(".py"):
            test_file = f"{test_file}.py"

        exit_code = run_specific_test(test_file)
    else:
        # Run all tests
        exit_code = run_tests()

    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")

    sys.exit(exit_code)
