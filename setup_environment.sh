#!/bin/bash

# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# Setup script for Isaac Lab with Piper Robot

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "Isaac Lab + Piper Robot - Environment Setup"
echo "================================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed!${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ Conda found${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Working directory: $SCRIPT_DIR${NC}"
echo ""

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo -e "${RED}Error: environment.yml not found!${NC}"
    exit 1
fi

# Check if isaacenv already exists
if conda env list | grep -q "^isaacenv "; then
    echo -e "${YELLOW}Warning: 'isaacenv' environment already exists${NC}"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n isaacenv -y
    else
        echo "Aborting setup."
        exit 0
    fi
fi

# Create conda environment
echo -e "${YELLOW}Creating conda environment 'isaacenv'...${NC}"
echo "This may take several minutes..."
conda env create -f environment.yml

# Activate environment
echo ""
echo -e "${YELLOW}Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate isaacenv

# Install Isaac Lab packages in editable mode
echo ""
echo -e "${YELLOW}Installing Isaac Lab packages in editable mode...${NC}"

if [ -d "source/isaaclab" ]; then
    echo "Installing isaaclab..."
    pip install -e ./source/isaaclab
else
    echo -e "${RED}Warning: source/isaaclab not found, skipping${NC}"
fi

if [ -d "source/isaaclab_assets" ]; then
    echo "Installing isaaclab_assets..."
    pip install -e ./source/isaaclab_assets
else
    echo -e "${YELLOW}Warning: source/isaaclab_assets not found, skipping${NC}"
fi

if [ -d "source/isaaclab_tasks" ]; then
    echo "Installing isaaclab_tasks..."
    pip install -e ./source/isaaclab_tasks
else
    echo -e "${YELLOW}Warning: source/isaaclab_tasks not found, skipping${NC}"
fi

if [ -d "source/isaaclab_rl" ]; then
    echo "Installing isaaclab_rl..."
    pip install -e ./source/isaaclab_rl
else
    echo -e "${YELLOW}Warning: source/isaaclab_rl not found, skipping${NC}"
fi

if [ -d "source/isaaclab_mimic" ]; then
    echo "Installing isaaclab_mimic..."
    pip install -e ./source/isaaclab_mimic
else
    echo -e "${YELLOW}Warning: source/isaaclab_mimic not found, skipping${NC}"
fi

# Install Piper SDK
if [ -d "piper_sdk" ]; then
    echo "Installing piper_sdk..."
    pip install -e ./piper_sdk
else
    echo -e "${YELLOW}Warning: piper_sdk not found, skipping${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "To activate the environment, run:"
echo -e "  ${YELLOW}conda activate isaacenv${NC}"
echo ""
echo "To test the installation, run:"
echo -e "  ${YELLOW}./isaaclab.sh -p piper_rl_project/demo_moves.py${NC}"
echo ""
echo -e "${YELLOW}Note: Make sure Isaac Sim is installed before running simulations.${NC}"
echo "Isaac Sim download: https://developer.nvidia.com/isaac-sim"
echo ""

