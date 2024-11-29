FROM python:3.9-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt
RUN useradd -m jupyter

# Copy the content of the local src directory to the working directory
COPY . .

# Change the ownership of the working directory to the jupyter user
RUN chown -R jupyter:jupyter /app
USER jupyter

# Start the port
EXPOSE 8888

# Run the application
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

