#!/bin/bash

# Update the package list
sudo apt-get update

# Install the required dependencies
sudo apt-get install -y \
    libopenblas-dev \
    libomp-dev \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libsndfile1 \
    ffmpeg

# Clean up the apt cache
sudo rm -rf /var/lib/apt/lists/*

echo "Dependencies installed successfully!"
