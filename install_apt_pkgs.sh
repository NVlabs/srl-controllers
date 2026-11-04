#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22

# ------------------------------------------------------------------------------
# Script: install_apt_packages.sh
# Description:
#   This script reads a list of apt packages with optional version constraints,
#   checks if they are installed, installs them if necessary, and verifies that
#   the installed version meets the given constraints.
#
# Usage:
#   ./install_apt_packages.sh [OPTIONS] <package_list_file>
#
# Options:
#   -s, --strict      Treat warnings as errors (return non-zero exit code on warnings)
#   -r, --reinstall   Force reinstall if the installed version differs from explicit install version
#   -d, --debug       Enable debug mode
#   -h, --help        Display this help message and exit
#
# Example:
#   ./install_apt_packages.sh apt-packages.txt
#   ./install_apt_packages.sh --strict apt-packages.txt
#   ./install_apt_packages.sh --reinstall apt-packages.txt
#
# Example Input File (apt-packages.txt):
#   libeigen3-dev>=3.4.0,<4.0.0
#
# Output:
#   - Installed versions are printed in green.
#   - Warnings about version mismatches are printed in yellow (or red in --strict mode).
#   - Errors (e.g., invalid package names) are printed in red.
#   - If --reinstall is provided, explicitly specified versions will be enforced.
#
# Notes:
#   - The script ensures version comparisons are done properly.
#   - Uses "dpkg --compare-versions" for comparisons.
#   - If a package is missing, it attempts to install it using "sudo apt install".
#   - If a package is not found in apt repositories, an error is displayed.
#   - If --reinstall is used, only the specified package is reinstalled without removing dependencies.
# ------------------------------------------------------------------------------

set -e

STRICT_MODE=false
REINSTALL_MODE=false
DEBUG_MODE=false
PACKAGE_LIST_FILE=""

# Function to display help message
print_help() {
    echo "Usage: $0 [OPTIONS] <package_list_file>"
    echo ""
    echo "Options:"
    echo "  -s, --strict      Treat warnings as errors (return non-zero exit code on warnings)"
    echo "  -r, --reinstall   Force reinstall if the installed version differs from explicit install version"
    echo "  -d, --debug       Enable debug mode"
    echo "  -h, --help        Display this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 apt-packages.txt"
    echo "  $0 --strict apt-packages.txt"
    echo "  $0 --reinstall apt-packages.txt"
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--strict)
            STRICT_MODE=true
            shift
            ;;
        -r|--reinstall)
            REINSTALL_MODE=true
            shift
            ;;
        -d|--debug)
            DEBUG_MODE=true
            shift
            ;;
        -h|--help)
            print_help
            ;;
        *)
            PACKAGE_LIST_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "$PACKAGE_LIST_FILE" ]]; then
    echo -e "\e[31mError: No package list file provided. Use -h for help.\e[0m"
    exit 1
fi

if [[ ! -f "$PACKAGE_LIST_FILE" ]]; then
    echo -e "\e[31mError: File '$PACKAGE_LIST_FILE' not found.\e[0m"
    exit 1
fi

# Color codes
RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
BLUE="\e[34m"
MAGENTA="\e[35m"
CYAN="\e[36m"
WHITE="\e[37m"
GRAY="\e[90m"
DEBUG=$GRAY
NC="\e[0m"  # No color

EXIT_STATUS=0

# Function to extract installed package version
get_installed_version() {
    local package="$1"
    local version
    version=$(dpkg-query -W -f='${Version}\n' "$package" 2>/dev/null | sed -E 's/^([0-9]+:)//' | sed -E 's/[-+~].*$//')
    if [[ -z "$version" ]]; then
        echo "NOT_INSTALLED"
    else
        echo "$version"
    fi
}

# Function to validate package existence
is_valid_package() {
    apt-cache show "$1" >/dev/null 2>&1
    return $?
}

# Function to compare versions with proper conditional logic
version_satisfies() {
    local v1="$1"
    local op="$2"
    local v2="$3"

    # Ensure versions have at least three components (major.minor.patch)
    v1=$(echo "$v1" | awk -F'.' '{printf "%d.%d.%d", $1, ($2 != "" ? $2 : 0), ($3 != "" ? $3 : 0)}')
    v2=$(echo "$v2" | awk -F'.' '{printf "%d.%d.%d", $1, ($2 != "" ? $2 : 0), ($3 != "" ? $3 : 0)}')

    case "$op" in
        "==") [[ "$v1" == "$v2" ]]; return $?
        ;;
        "!=") [[ "$v1" != "$v2" ]]; return $?
        ;;
        "<") dpkg --compare-versions "$v1" "lt" "$v2"; return $?
        ;;
        "<=") dpkg --compare-versions "$v1" "le" "$v2"; return $?
        ;;
        ">") dpkg --compare-versions "$v2" "lt" "$v1"; return $?
        ;;
        ">=") dpkg --compare-versions "$v2" "le" "$v1"; return $?
        ;;
        *) echo -e "${RED}Unknown operator: $op${NC}"; return 1;;
    esac
}

# Check if the file ends with a newline
if [[ -n $(tail -c1 "$PACKAGE_LIST_FILE" | tr -d '\n') ]]; then
    echo -e "${RED}Error: The package list file '$PACKAGE_LIST_FILE' does not end with a newline.${NC}"
    echo -e "${RED}Please add a newline at the end of the file and try again.${NC}"
    exit 1
fi

while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue  # Skip empty lines and comments

    # Extract PACKAGE including explicit version if present (everything before the first comma, if any)
    PACKAGE=$(echo "$line" | grep -o -E '^[^,]+')

    # Extract EXPLICIT_VERSION if present
    EXPLICIT_VERSION=$(echo "$PACKAGE" | grep -o -E '=[0-9]+(\.[0-9]+)*' | cut -d '=' -f2)

    # Extract PACKAGE name without explicit version
    PACKAGE=$(echo "$PACKAGE" | grep -o -E '^[^=]+')

    # Extract CONSTRAINTS (everything after the first comma, if present)
    CONSTRAINTS=$(echo "$line" | grep -o -E ',.*' | cut -c2- | tr '\n' ',' | sed 's/,$//')

    # Default display for constraints (handles empty case)
    CONSTRAINTS_DISPLAY=${CONSTRAINTS:-"Any"}

    # Print extracted values (for debugging)
    if [[ "$DEBUG_MODE" == "true" ]]; then
        echo -e "${DEBUG}PACKAGE: $PACKAGE${NC}"
        echo -e "${DEBUG}EXPLICIT_VERSION: ${EXPLICIT_VERSION:-None}${NC}"
        echo -e "${DEBUG}CONSTRAINTS: $CONSTRAINTS_DISPLAY${NC}"
    fi

    if ! is_valid_package "$PACKAGE"; then
        echo -e "${RED}Error: Package '$PACKAGE' does not exist in the apt repositories. Skipping.${NC}"
        EXIT_STATUS=1
        continue
    fi

    INSTALLED_VERSION=$(get_installed_version "$PACKAGE")

    if version_satisfies "$INSTALLED_VERSION" "==" "$EXPLICIT_VERSION"; then
        VERSIONS_ARE_EQUAL=true
    else
        VERSIONS_ARE_EQUAL=false
    fi

    if [[ "$INSTALLED_VERSION" != "NOT_INSTALLED" && "$REINSTALL_MODE" == "true" && "$EXPLICIT_VERSION" != "" && "$VERSIONS_ARE_EQUAL" == "false" ]]; then
        echo -e "${YELLOW}Reinstalling $PACKAGE: installing version $EXPLICIT_VERSION over $INSTALLED_VERSION.${NC}"
        sudo apt install -y --allow-downgrades "$PACKAGE=$EXPLICIT_VERSION"
        INSTALLED_VERSION=$(get_installed_version "$PACKAGE")
    fi

    if [[ "$INSTALLED_VERSION" == "NOT_INSTALLED" ]]; then
        INSTALL_CMD="sudo apt install -y $PACKAGE"
        [[ -n "$EXPLICIT_VERSION" ]] && INSTALL_CMD+="=$EXPLICIT_VERSION"
        echo -e "${GREEN}Package $PACKAGE is not installed. Installing...${NC}"
        if ! eval "$INSTALL_CMD"; then
            echo -e "${YELLOW}Warning: Failed to install $PACKAGE${NC}"
            EXIT_STATUS=1
            continue
        fi
        INSTALLED_VERSION=$(get_installed_version "$PACKAGE")
    fi

    echo -e "${GREEN}Package: $PACKAGE${NC}"
    echo -e "  ${GREEN}Installed Version: $INSTALLED_VERSION${NC}"
    echo -e "  ${GREEN}Required Version: $CONSTRAINTS_DISPLAY${NC}"

    # Check version constraints
    for CONSTRAINT in $(echo "$CONSTRAINTS" | tr ',' '\n'); do
        OPERATOR=$(echo "$CONSTRAINT" | grep -o -E '^[<>]=?|==')
        VERSION=$(echo "$CONSTRAINT" | grep -o -E '[0-9]+(\.[0-9]+)*$')

        if ! version_satisfies "$INSTALLED_VERSION" "$OPERATOR" "$VERSION"; then
            if $STRICT_MODE; then
                echo -e "  ${RED}Error: Installed version does not meet requirement: $CONSTRAINT${NC}"
                EXIT_STATUS=1
            else
                echo -e "  ${YELLOW}Warning: Installed version does not meet requirement: $CONSTRAINT${NC}"
            fi
        fi
    done

done < "$PACKAGE_LIST_FILE"

exit $EXIT_STATUS
