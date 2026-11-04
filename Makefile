# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# Makefile for common dev-ops tasks.
#
# Run `make help` for basic help message.

# Set default shell for all targets
SHELL := /bin/bash

# Set virtual environment path
VENV_PATH ?= $(CURDIR)/venv/3.12-cpp-controllers

# Set build options
override BUILD_DIR := $(CURDIR)/_build

# Set check options
CHECK_LICENSE_HEADERS := true
CHECK_PUBLIC_API := true

CHECK_CPPLINT := true
CPPLINT_DIRS := $(CURDIR)/examples $(CURDIR)/include $(CURDIR)/src $(CURDIR)/tests $(CURDIR)/tools

CHECK_BLACK := true
BLACK_DIRS := $(CURDIR)/python

CHECK_FLAKE8 := true
FLAKE8_DIRS := $(CURDIR)/python

CHECK_MYPY := true
MYPY_DIRS := $(CURDIR)/python/src/nvidia/srl $(CURDIR)/python/tests

# Set format options
FORMAT_CLANG := true
FORMAT_BLACK := true
FORMAT_ISORT := true
ISORT_DIRS := $(CURDIR)

# Set docs variables
DOCS_DOXYGEN := true

# Set test and coverage variables
TEST_CPP := true
TEST_PYTHON := true
COVERAGE_CPP := true
COVERAGE_PYTHON := true

# - Set python test and coverage variables
TEST_PY_DIR := $(CURDIR)/python/tests
COVERAGE_PY_SRCDIR := $(CURDIR)/python/src
COVERAGE_PY_BUILDDIR := $(BUILD_DIR)/coverage/py
PYTEST_XDIST_ARGS ?= $(shell python3 -c "import importlib.util; print('-n auto' if importlib.util.find_spec('xdist') else '')")
PYTHONPATH_TEST := $(CURDIR)/python/src:$${PYTHONPATH}

# Set package variables
PACKAGE_CPP := true
PACKAGE_PYTHON := true

# Set check target dependencies
CHECK_DEPS =

ifeq ($(CHECK_PUBLIC_API),true)
	CHECK_DEPS += check_public_api
endif

ifeq ($(CHECK_LICENSE_HEADERS),true)
	CHECK_DEPS += check_license_headers
endif

ifeq ($(CHECK_CPPLINT),true)
	CHECK_DEPS += check_cpplint
endif

ifeq ($(CHECK_BLACK),true)
	CHECK_DEPS += check_black
endif

ifeq ($(CHECK_FLAKE8),true)
	CHECK_DEPS += check_flake8
endif

ifeq ($(CHECK_MYPY),true)
	CHECK_DEPS += check_mypy
endif

# Set format target dependencies
FORMAT_DEPS =

ifeq ($(FORMAT_CLANG),true)
	FORMAT_DEPS += format_clang
endif

ifeq ($(FORMAT_ISORT),true)
	FORMAT_DEPS += format_isort
endif

ifeq ($(FORMAT_BLACK),true)
	FORMAT_DEPS += format_black
endif

# Set test target dependencies
TEST_DEPS =

ifeq ($(TEST_CPP),true)
	TEST_DEPS += test_cpp
endif

ifeq ($(TEST_PYTHON),true)
	TEST_DEPS += test_py
endif

# Set coverage target dependencies
COVERAGE_DEPS =

ifeq ($(TEST_CPP),true)
	ifeq ($(COVERAGE_CPP),true)
		COVERAGE_DEPS += coverage_cpp
	endif
endif

ifeq ($(TEST_PYTHON),true)
	ifeq ($(COVERAGE_PYTHON),true)
		COVERAGE_DEPS += coverage_py
	endif
endif

# Set docs target dependencies
DOCS_DEPS =

ifeq ($(DOCS_DOXYGEN),true)
	DOCS_DEPS += docs_doxygen
endif

# Set package target dependencies
PACKAGE_DEPS =

ifeq ($(PACKAGE_CPP),true)
	PACKAGE_DEPS += package_cpp
endif

ifeq ($(PACKAGE_PYTHON),true)
	PACKAGE_DEPS += package_py
endif

# Declare phony targets
.PHONY: help sudo \
				venv venv_install_e venv_install_ci venv_install_dev \
				init \
        build \
        check check_public_api check_license_headers check_cpplint check_black check_flake8 check_mypy \
        docs docs_doxygen \
        format format_clang format_isort format_black \
        test test_cpp test_py \
        coverage coverage_cpp coverage_py \
        package package_cpp package_py \
				install \
        clean clean_all

# Default target: Display a basic help message
help:
	@echo "Make targets:"
	@echo ""
	@echo "  venv             Create a Python virtual environment."
	@echo "  venv_install_e   Create a Python virtual environment and install the editable project and build dependencies."
	@echo "  venv_install_ci  Create a Python virtual environment and install the editable project, build, and CI dependencies."
	@echo "  venv_install_dev Compatibility alias for venv_install_ci."
	@echo "  apt_install      Install apt dependencies."
	@echo "  init             Initialize the project by creating the virtual environment and installing the project into it."
	@echo "  build            Build the project."
	@echo "  check            Run code checks."
	@echo "  docs             Generate documentation."
	@echo "  format           Auto-format code."
	@echo "  test             Run all tests."
	@echo "  coverage         Generate a coverage reports."
	@echo "  install          Install the build artifacts into the install directory."
	@echo "  package          Generate a debian and pip packages."
	@echo "  clean            Delete the build directory."
	@echo "  clean_all        Delete the build directory and virtual environment."
	@echo ""
	@echo "Use 'make <target>' to execute a target."
	@echo ""
	@echo "Make useful options:"
	@echo ""
	@echo "  -B    Unconditionally make all targets."
	@echo "  -n    Don't actually run any recipe; just print them."
	@echo "  -h    Print full make help and exit."

# Directory dependency rule
$(VENV_PATH):
	mkdir -p $(VENV_PATH)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Install apt dependencies
APT_STAMP = $(CURDIR)/.apt_install.stamp

$(APT_STAMP): $(CURDIR)/apt_pkg_list.txt $(CURDIR)/apt_pkg_list_ci.txt
	sudo apt update && \
	sudo $(CURDIR)/install_apt_pkgs.sh -r -s $(CURDIR)/apt_pkg_list.txt && \
	sudo $(CURDIR)/install_apt_pkgs.sh -r -s $(CURDIR)/apt_pkg_list_ci.txt && \
	touch $@

apt_install: $(APT_STAMP)

# Create a Python virtual environment
venv: $(VENV_PATH)
	python3 -m venv $(VENV_PATH) && \
	source $(VENV_PATH)/bin/activate && pip install -U pip

# Create a Python virtual environment and install the editable project and build dependencies
venv_install_e: venv
	source $(VENV_PATH)/bin/activate && pip install -e '.[build]'

# Create a Python virtual environment and install the project with build and CI dependencies
venv_install_ci: venv
	source $(VENV_PATH)/bin/activate && pip install -e '.[build,ci]'

# Compatibility alias for the former dev extra
venv_install_dev: venv_install_ci

# Initialize the project by creating the virtual environment and installing dependencies
init: venv_install_dev

# Build the project
build: $(BUILD_DIR) apt_install
	cmake -B$(BUILD_DIR) -S$(CURDIR) -DDEVEL=ON && \
	make -C $(BUILD_DIR)

# Run the checks
check: $(CHECK_DEPS)

check_license_headers:
	python3 $(CURDIR)/tools/check_license_headers.py --self-test
	python3 $(CURDIR)/tools/check_license_headers.py

check_public_api:
	python3 tools/generate_public_blocks.py --check
	python3 tools/generate_equation_api.py --check
	python3 tools/verify_cuda_python_tests.py --self-test

check_cpplint:
	python3 -m cpplint --recursive --quiet $(CPPLINT_DIRS)

check_black:
	python3 -m black --config $(CURDIR)/pyproject.toml --check --diff $(BLACK_DIRS)

check_flake8:
	python3 -m flake8 --toml-config $(CURDIR)/pyproject.toml $(FLAKE8_DIRS)

check_mypy:
	@set -e; \
	cd python/src && \
	for dir in $(MYPY_DIRS); \
	do \
		python3 -m mypy --config-file=$(CURDIR)/pyproject.toml --namespace-packages --explicit-package-bases $$dir; \
	done

# Run formatter
format: $(FORMAT_DEPS)

format_clang:
		find $(CURDIR)/include $(CURDIR)/examples $(CURDIR)/src $(CURDIR)/tests $(CURDIR)/tools \( -name "*.hpp" -o -name "*.cpp" -o -name "*.tpp" \) -print0 | xargs -0 clang-format -i

format_isort:
	python3 -m isort --sp $(CURDIR)/pyproject.toml $(ISORT_DIRS)

format_black:
	python3 -m black --config $(CURDIR)/pyproject.toml $(BLACK_DIRS)

# Run the tests
test: $(TEST_DEPS)

test_cpp:
	make -C $(BUILD_DIR) test

test_py:
	PYTHONPATH="$(PYTHONPATH_TEST)" python3 -m pytest $(PYTEST_XDIST_ARGS) -c $(CURDIR)/pyproject.toml $(TEST_PY_DIR)

# Generate the coverage report
coverage: $(COVERAGE_DEPS)

coverage_cpp:
	make -C $(BUILD_DIR) coverage

coverage_py:
	PYTHONPATH="$(PYTHONPATH_TEST)" python3 -m pytest $(PYTEST_XDIST_ARGS) -c $(CURDIR)/pyproject.toml --cov-report=term --cov-report=xml:$(COVERAGE_PY_BUILDDIR)/coverage.xml --cov-report=html:$(COVERAGE_PY_BUILDDIR)/html/ --cov=$(COVERAGE_PY_SRCDIR) $(TEST_PY_DIR)

# Generate the documentation
docs: $(DOCS_DEPS)

docs_doxygen:
	make -C $(BUILD_DIR) docs

# Generate a Python package
package: $(PACKAGE_DEPS)

package_cpp:
	make -C $(BUILD_DIR) package

package_py:
	python3 -m build --outdir $(BUILD_DIR)/packages

# Install the build artifacts into the install directory
install:
	make -C $(BUILD_DIR) install

# Delete the build directories
clean:
	rm -rf $(BUILD_DIR)

# Delete the build directories and virtual environment
clean_all: clean
	rm -rf $(VENV_PATH)
