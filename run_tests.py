#!/usr/bin/env python3
"""
Test runner script for the audio transcription service.

This script provides a convenient way to run different types of tests
with various configurations and options.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    # Add test markers for unit tests
    cmd.extend(["-m", "unit"])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run integration tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    # Add test markers for integration tests
    cmd.extend(["-m", "integration"])
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    return run_command(cmd, "Integration Tests")


def run_all_tests(verbose: bool = False, coverage: bool = False, slow: bool = False) -> bool:
    """Run all tests."""
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    if not slow:
        cmd.extend(["-m", "not slow"])
    
    return run_command(cmd, "All Tests")


def run_specific_test(test_path: str, verbose: bool = False) -> bool:
    """Run a specific test file or test function."""
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    return run_command(cmd, f"Specific Test: {test_path}")


def run_linting() -> bool:
    """Run code linting."""
    success = True
    
    # Run ruff
    if not run_command(["python", "-m", "ruff", "check", "src/", "tests/"], "Ruff Linting"):
        success = False
    
    # Run black check
    if not run_command(["python", "-m", "black", "--check", "src/", "tests/"], "Black Formatting Check"):
        success = False
    
    # Run mypy
    if not run_command(["python", "-m", "mypy", "src/"], "MyPy Type Checking"):
        success = False
    
    return success


def run_formatting() -> bool:
    """Run code formatting."""
    success = True
    
    # Run black
    if not run_command(["python", "-m", "black", "src/", "tests/"], "Black Formatting"):
        success = False
    
    # Run ruff fix
    if not run_command(["python", "-m", "ruff", "check", "--fix", "src/", "tests/"], "Ruff Auto-fix"):
        success = False
    
    return success


def run_security_check() -> bool:
    """Run security checks."""
    cmd = ["python", "-m", "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"]
    return run_command(cmd, "Security Check (Bandit)")


def run_performance_tests() -> bool:
    """Run performance tests."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "performance", "-v"]
    return run_command(cmd, "Performance Tests")


def run_quick_tests() -> bool:
    """Run quick tests (unit tests only, no slow tests)."""
    cmd = ["python", "-m", "pytest", "tests/", "-m", "unit and not slow", "-v"]
    return run_command(cmd, "Quick Tests")


def run_ci_tests() -> bool:
    """Run tests as they would be run in CI."""
    success = True
    
    print("Running CI test suite...")
    
    # Run linting
    if not run_linting():
        success = False
    
    # Run unit tests with coverage
    if not run_unit_tests(verbose=True, coverage=True):
        success = False
    
    # Run integration tests (without slow tests)
    if not run_integration_tests(verbose=True):
        success = False
    
    # Run security check
    if not run_security_check():
        success = False
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for audio transcription service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py --all
  
  # Run unit tests only
  python run_tests.py --unit
  
  # Run integration tests
  python run_tests.py --integration
  
  # Run specific test file
  python run_tests.py --test tests/test_transcriber.py
  
  # Run with coverage
  python run_tests.py --all --coverage
  
  # Run linting
  python run_tests.py --lint
  
  # Run formatting
  python run_tests.py --format
  
  # Run CI tests
  python run_tests.py --ci
  
  # Run quick tests (unit tests, no slow tests)
  python run_tests.py --quick
        """
    )
    
    # Test selection
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--test", type=str, help="Run specific test file or function")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (unit, no slow)")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    
    # Test options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    
    # Code quality
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--format", action="store_true", help="Run code formatting")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    
    # CI
    parser.add_argument("--ci", action="store_true", help="Run CI test suite")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    success = True
    
    # Run selected tests/checks
    if args.ci:
        success = run_ci_tests()
    elif args.all:
        success = run_all_tests(args.verbose, args.coverage, args.slow)
    elif args.unit:
        success = run_unit_tests(args.verbose, args.coverage)
    elif args.integration:
        success = run_integration_tests(args.verbose, args.coverage)
    elif args.test:
        success = run_specific_test(args.test, args.verbose)
    elif args.quick:
        success = run_quick_tests()
    elif args.performance:
        success = run_performance_tests()
    elif args.lint:
        success = run_linting()
    elif args.format:
        success = run_formatting()
    elif args.security:
        success = run_security_check()
    else:
        parser.print_help()
        return 0
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests/checks completed successfully!")
        return 0
    else:
        print("❌ Some tests/checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
