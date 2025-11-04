#!/bin/bash

echo "=========================================="
echo "Running Hello World Service Test Suite"
echo "=========================================="
echo ""

echo "Step 1: Cleaning previous build artifacts..."
mvn clean

echo ""
echo "Step 2: Running all tests..."
mvn test

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed successfully!"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "✗ Tests failed. Please check the output above."
    echo "=========================================="
    exit 1
fi
