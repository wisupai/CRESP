#!/bin/bash

set -e

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Installation directory
INSTALL_DIR="/usr/local/bin"
if [ "$(uname)" = "Darwin" ]; then
    # On macOS, check if /usr/local/bin is writable, otherwise use ~/.local/bin
    if [ ! -w "/usr/local/bin" ]; then
        INSTALL_DIR="${HOME}/.local/bin"
        mkdir -p "${INSTALL_DIR}"
    fi
elif [ "$(uname)" = "Linux" ]; then
    # On Linux, use ~/.local/bin if not root
    if [ "$EUID" -ne 0 ]; then
        INSTALL_DIR="${HOME}/.local/bin"
        mkdir -p "${INSTALL_DIR}"
    fi
fi

# Determine the latest version if not specified
VERSION=${1:-$(curl -s https://api.github.com/repos/wisupai/CRESP/releases/latest | grep -o '"tag_name": "v[^"]*"' | grep -o 'v[^"]*' 2>/dev/null || echo "v0.1.0-dev.1")}

echo -e "${BLUE}Downloading CRESP ${VERSION}...${NC}"

# Determine architecture
PLATFORM="unknown"
ARCH=$(uname -m)

case "$(uname)" in
    "Darwin")
        PLATFORM="darwin"
        if [ "$ARCH" = "arm64" ]; then
            ARCH="aarch64"
        fi
        ;;
    "Linux")
        PLATFORM="linux"
        ;;
    "MINGW"*|"MSYS"*|"CYGWIN"*)
        PLATFORM="windows"
        INSTALL_DIR="."
        EXT=".exe"
        ;;
    *)
        echo -e "${RED}Unsupported operating system: $(uname)${NC}"
        exit 1
        ;;
esac

# Download URL
DOWNLOAD_URL="https://github.com/wisupai/CRESP/releases/download/${VERSION}/cresp-${PLATFORM}-${ARCH}.tar.gz"
if [ "$PLATFORM" = "windows" ]; then
    DOWNLOAD_URL="https://github.com/wisupai/CRESP/releases/download/${VERSION}/cresp-${PLATFORM}-${ARCH}.zip"
fi

# Create a temporary directory for downloading and extraction
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# Download the binary
echo -e "${BLUE}Downloading from ${DOWNLOAD_URL}...${NC}"
if command -v curl &> /dev/null; then
    curl -L "$DOWNLOAD_URL" -o "cresp-package"
elif command -v wget &> /dev/null; then
    wget -O "cresp-package" "$DOWNLOAD_URL"
else
    echo -e "${RED}Neither curl nor wget is available. Please install one of them and try again.${NC}"
    exit 1
fi

# Extract the binary
echo -e "${BLUE}Extracting...${NC}"
if [ "$PLATFORM" = "windows" ]; then
    unzip "cresp-package"
else
    tar xzf "cresp-package"
fi

# Move the binary to the installation directory
echo -e "${BLUE}Installing CRESP to ${INSTALL_DIR}...${NC}"
if [ "$PLATFORM" = "windows" ]; then
    mv "cresp${EXT}" "${INSTALL_DIR}/cresp${EXT}"
else
    mv "cresp" "${INSTALL_DIR}/cresp"
    chmod +x "${INSTALL_DIR}/cresp"
fi

# Clean up the temporary directory
rm -rf "$TMP_DIR"

echo -e "${GREEN}CRESP ${VERSION} has been installed to ${INSTALL_DIR}/cresp${EXT:-}${NC}"

# Check if directory is in PATH
PATH_UPDATED=0
if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
    echo -e "${YELLOW}Adding ${INSTALL_DIR} to your PATH for current session...${NC}"
    export PATH="$PATH:${INSTALL_DIR}"
    PATH_UPDATED=1
    
    # Detect shell and update config file
    SHELL_CONFIG=""
    if [ -n "$BASH_VERSION" ]; then
        if [ -f "$HOME/.bashrc" ]; then
            SHELL_CONFIG="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            SHELL_CONFIG="$HOME/.bash_profile"
        fi
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi
    
    if [ -n "$SHELL_CONFIG" ]; then
        if ! grep -q "export PATH=.*${INSTALL_DIR}" "$SHELL_CONFIG"; then
            echo -e "${YELLOW}Updating ${SHELL_CONFIG} to add ${INSTALL_DIR} to PATH permanently...${NC}"
            echo "" >> "$SHELL_CONFIG"
            echo "# Added by CRESP installer" >> "$SHELL_CONFIG"
            echo "export PATH=\"\$PATH:${INSTALL_DIR}\"" >> "$SHELL_CONFIG"
            echo -e "${GREEN}PATH has been updated in ${SHELL_CONFIG}.${NC}"
            echo -e "${YELLOW}To apply changes to current session, run:${NC}"
            echo -e "${YELLOW}  source ${SHELL_CONFIG}${NC}"
        else
            echo -e "${GREEN}${INSTALL_DIR} is already in your PATH configuration.${NC}"
        fi
    else
        echo -e "${YELLOW}Could not detect shell configuration file.${NC}"
        echo -e "${YELLOW}Please add the following to your shell configuration file:${NC}"
        echo -e "${YELLOW}  export PATH=\"\$PATH:${INSTALL_DIR}\"${NC}"
    fi
fi

# Verify installation
if command -v cresp &> /dev/null || [ -x "${INSTALL_DIR}/cresp${EXT:-}" ]; then
    echo -e "${GREEN}Installation successful! You can now use CRESP by running 'cresp'.${NC}"
    echo -e "${GREEN}Run 'cresp --help' to see available commands.${NC}"
    
    # Show PATH note only if we didn't update it
    if [ "$PATH_UPDATED" -eq 0 ]; then
        echo -e "${GREEN}${INSTALL_DIR} is already in your PATH.${NC}"
    fi
    
    # Print version to verify executable works
    echo -e "${BLUE}Installed version:${NC}"
    if [ "$PATH_UPDATED" -eq 1 ]; then
        # Use full path if PATH was just updated (safer)
        "${INSTALL_DIR}/cresp${EXT:-}" --version
    else
        cresp --version
    fi
else
    echo -e "${RED}Installation may have completed, but 'cresp' command is not available.${NC}"
    echo -e "${RED}Please ensure ${INSTALL_DIR} is in your PATH and try again.${NC}"
    echo -e "${YELLOW}You can run the command directly using:${NC}"
    echo -e "${YELLOW}  ${INSTALL_DIR}/cresp${EXT:-}${NC}"
fi

# For sourced scripts to work without breaking the terminal session
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo -e "${YELLOW}PATH will be updated for current session.${NC}"
fi 