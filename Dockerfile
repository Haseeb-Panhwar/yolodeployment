FROM python:3.11-slim

# Install required system packages for OpenCV and image handling
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for the Flask app
EXPOSE 5000

# Start the Flask app with Gunicorn
# CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers=4", "--worker-class=gevent", "--timeout=10000"]
CMD ["python", "app.py"]
