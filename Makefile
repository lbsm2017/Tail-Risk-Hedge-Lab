# Makefile for Tail-Risk Hedge Lab
# Author: L.Bassetti
#
# This Makefile provides convenient commands for running the backtesting
# framework and executing the test suite.

.PHONY: help run tests clean

# Default target - show help
help:
	@echo ============================================================
	@echo Tail-Risk Hedging Backtesting Framework
	@echo ============================================================
	@echo.
	@echo Available commands:
	@echo.
	@echo   make run       Run the complete backtesting pipeline
	@echo   make tests     Run all unit and integration tests
	@echo   make clean     Remove cached files and test artifacts
	@echo   make help      Show this help message
	@echo.
	@echo ============================================================

# Run the main backtesting pipeline
# Executes main.py which orchestrates data download, regime detection,
# individual hedge analysis, portfolio optimization, and report generation
run:
	@echo Running backtesting pipeline...
	python main.py

# Run all tests
# Executes the comprehensive test suite including unit tests for main.py,
# engine.py, and integration tests for the complete pipeline
tests:
	@echo Running test suite...
	python tests/all_tests.py

# Clean up cached files and test outputs
# Removes Python cache files, test reports, and temporary artifacts
clean:
	@echo Cleaning up cached files...
	@if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	@if exist src\__pycache__ rmdir /s /q src\__pycache__
	@if exist src\backtester\__pycache__ rmdir /s /q src\backtester\__pycache__
	@if exist src\data\__pycache__ rmdir /s /q src\data\__pycache__
	@if exist src\hypothesis\__pycache__ rmdir /s /q src\hypothesis\__pycache__
	@if exist src\metrics\__pycache__ rmdir /s /q src\metrics\__pycache__
	@if exist src\optimization\__pycache__ rmdir /s /q src\optimization\__pycache__
	@if exist src\regime\__pycache__ rmdir /s /q src\regime\__pycache__
	@if exist src\reporting\__pycache__ rmdir /s /q src\reporting\__pycache__
	@if exist output\test_report.html del /q output\test_report.html
	@echo Clean complete!
