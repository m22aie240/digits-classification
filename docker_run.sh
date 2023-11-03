#!/bin/bash

# Build the Docker image
docker build -t digits -f docker/Dockerfile .

# Run the Docker container and mount the models directory
docker run -v $(pwd)/models:/digits/models digits
