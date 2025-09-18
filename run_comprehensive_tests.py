#!/usr/bin/env python3
"""
Comprehensive test runner for ACME AI/ML Model Evaluation System.
Executes the complete test suite following the validation plan format.
"""

import sys
import os
import subprocess
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def print_header(title):
    """Print a formatted header."""
    print(f"\n🧪 {title}")
    print("=" * 60)


def run_test_category(category_name, test_files):
    """Run a category of tests and return the result."""
    print(f"\n📋 {category_name}")
    print("-" * 40)
    
    all_passed = True
    for test_file in test_files:
        if os.path.exists(test_file):
            cmd = [
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
            ]
            print(f"Running: {test_file}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            
            if result.returncode == 0:
                print(f"✅ {os.path.basename(test_file)} - PASSED")
            else:
                print(f"❌ {os.path.basename(test_file)} - FAILED")
                print(result.stdout)
                print(result.stderr)
                all_passed = False
        else:
            print(f"⚠️  {test_file} - NOT FOUND (skipping)")
    
    return all_passed


def main():
    """Execute comprehensive validation plan tests."""
    print_header("ACME AI/ML Model Evaluation System - Comprehensive Testing")
    print("Following the structured validation plan format")
    
    start_time = time.time()
    all_categories_passed = True
    
    # Test categories following the validation plan format
    test_categories = [
        ("CLI Layer Tests", [
            "tests/test_cli_app.py",
            "tests/test_run_install.py",
            "tests/test_run_test.py"
        ]),
        ("Routing Layer Tests", [
            "tests/test_routing_layer.py"
        ]),
        ("Handler Tests", [
            "tests/test_url_handlers.py"
        ]),
        ("RepoContext Tests", [
            "tests/test_repo_context_class.py"
        ]),
        ("Core Metrics Tests", [
            "tests/test_base_metric.py",
            "tests/test_community_rating_metric.py"
        ]),
        ("Extended Metrics Tests", [
            "tests/test_extended_metrics.py"
        ]),
        ("MetricEval Comprehensive Tests", [
            "tests/test_metric_eval.py"
        ]),
        ("Integration & System Tests", [
            "tests/test_integration.py"
        ])
    ]
    
    # Execute each test category
    for category_name, test_files in test_categories:
        category_passed = run_test_category(category_name, test_files)
        if not category_passed:
            all_categories_passed = False
    
    # Summary
    end_time = time.time()
    execution_time = end_time - start_time
    
    print_header("Test Execution Summary")
    print(f"Total execution time: {execution_time:.2f} seconds")
    
    if all_categories_passed:
        print("✅ All test categories completed successfully!")
        print("🎉 System validation PASSED - Ready for production!")
        return 0
    else:
        print("❌ Some test categories failed")
        print("🔧 Please review failures and fix issues")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
