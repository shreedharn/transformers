#!/usr/bin/env python3
"""
Test runner for py-improve-md comprehensive test suite.

This script provides various ways to run the test suite with different
levels of detail and coverage reporting.
"""

import sys
import unittest
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_all_tests(verbosity=2, pattern="test_*.py"):
    """Run all tests with specified verbosity and pattern."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern=pattern)

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_category_tests(category, verbosity=2):
    """Run tests for specific detector category."""
    category_map = {
        'inline_math': 'TestInlineMathDetectors',
        'display_math': 'TestDisplayMathDetectors',
        'list_formatting': 'TestListFormattingDetectors',
        'alignment': 'TestAlignmentDetectors',
        'structural': 'TestStructuralDetectors',
        'syntax': 'TestSyntaxDetectors',
        'cli': 'TestCLIFunctionality',
        'edge_cases': 'TestEdgeCasesAndErrorHandling',
        'performance': 'TestPerformanceBenchmarks'
    }

    if category not in category_map:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(category_map.keys())}")
        return False

    import test_py_improve_md

    suite = unittest.TestSuite()
    test_class = getattr(test_py_improve_md, category_map[category])

    for method_name in dir(test_class):
        if method_name.startswith('test_'):
            suite.addTest(test_class(method_name))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Test runner for py-improve-md')
    parser.add_argument('--category', '-c', help='Run tests for specific category')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    parser.add_argument('--pattern', '-p', default='test_*.py', help='Test file pattern')

    args = parser.parse_args()

    # Set verbosity level
    verbosity = 1  # Default
    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 0

    success = True

    if args.category:
        success = run_category_tests(args.category, verbosity)
    else:
        success = run_all_tests(verbosity, args.pattern)

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())