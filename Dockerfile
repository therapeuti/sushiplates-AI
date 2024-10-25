# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt  # Add --no-cache-dir to avoid caching

# Copy application code and model files
COPY . .

# Make port 8080 available
EXPOSE 8080

# Run the application
CMD ["python", "app-fast.py"]
