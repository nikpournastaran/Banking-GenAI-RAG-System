#!/usr/bin/env python3
"""
Environment check script to verify configuration before starting the main application.
This helps detect common issues without modifying main.py.
"""

import os
import sys
import importlib
import traceback


def check_dependencies():
    """Check that all required dependencies can be imported"""
    required_libs = [
        "fastapi",
        "uvicorn",
        "langchain",
        "langchain_openai",
        "langchain_community",
        "langchain_text_splitters",
        "faiss",
        "openai",
        "dotenv"
    ]

    issues = []

    for lib in required_libs:
        try:
            importlib.import_module(lib.replace("-", "_"))
            print(f"✅ {lib} successfully imported")
        except ImportError as e:
            issues.append(f"❌ Failed to import {lib}: {str(e)}")

    return issues


def check_env_variables():
    """Check that required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["ADMIN_PASSWORD"]

    issues = []

    for var in required_vars:
        if not os.environ.get(var):
            issues.append(f"❌ Required environment variable {var} is not set")
        else:
            # Don't print actual values of sensitive variables
            print(f"✅ Environment variable {var} is set")

    for var in optional_vars:
        if not os.environ.get(var):
            print(f"⚠️ Optional environment variable {var} is not set")
        else:
            print(f"✅ Environment variable {var} is set")

    return issues


def check_directories():
    """Check that required directories exist and are accessible"""
    required_dirs = ["/data"]
    issues = []

    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created directory {directory}")
            except Exception as e:
                issues.append(f"❌ Failed to create directory {directory}: {str(e)}")
        else:
            # Check if writable
            if not os.access(directory, os.W_OK):
                issues.append(f"❌ Directory {directory} exists but is not writable")
            else:
                print(f"✅ Directory {directory} exists and is writable")

    return issues


def check_index():
    """Check if the FAISS index exists"""
    index_paths = [
        "/data/index.faiss",
        "./index/index.faiss"
    ]

    for path in index_paths:
        if os.path.exists(path):
            print(f"✅ Index found at {path}")
            # Check size
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   Index size: {size_mb:.2f} MB")
            return []

    return ["❌ FAISS index not found in any expected location"]


def main():
    """Run all checks and report issues"""
    print("\n===== Environment Check Report =====\n")

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check if running on Render
    is_render = os.environ.get("RENDER") == "true"
    print(f"Running on Render: {'Yes' if is_render else 'No'}")

    # Run all checks
    dependency_issues = check_dependencies()
    env_issues = check_env_variables()
    directory_issues = check_directories()
    index_issues = check_index()

    all_issues = dependency_issues + env_issues + directory_issues + index_issues

    if all_issues:
        print("\n⚠️ The following issues were detected:\n")
        for issue in all_issues:
            print(f"  {issue}")
        print("\nSome issues may prevent the application from running correctly.")
        if len(dependency_issues) > 0:
            print("Dependency issues need to be resolved before starting the application.")
            sys.exit(1)
    else:
        print("\n✅ All checks passed. Environment appears to be configured correctly.")

    print("\n======================================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error during environment check: {str(e)}")
        traceback.print_exc()
        sys.exit(1)