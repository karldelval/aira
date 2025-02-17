FROM ubuntu:latest

# Update package lists
RUN apt-get update

# Install gfortran and other dependencies
RUN apt-get install -y gfortran

# Copy your application code into the image
COPY . /app

# Set the working directory
WORKDIR /app

# Specify the command to run when the container starts
CMD ["main.py"]