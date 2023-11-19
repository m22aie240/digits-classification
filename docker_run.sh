#!/bin/bash

# Build the Docker image
docker build -t digits -f docker/Dockerfile .

# Run the Docker container and mount the models directory
echo $(pwd)
docker run -p 5002:5002 digits 
#-v $(pwd)/models:/digits/models digits
