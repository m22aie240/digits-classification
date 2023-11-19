# Use an official Python runtime as a parent image
FROM python:3.9.17

# Set the working directory in the container
WORKDIR /digits

# Copy just the requirements file into the container
COPY requirements.txt /digits/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . /digits

# Make port 5002 available to the world outside this container
EXPOSE 5002

# Define environment variable
ENV FLASK_APP=api/app.py

# Run flask when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=5002"]

