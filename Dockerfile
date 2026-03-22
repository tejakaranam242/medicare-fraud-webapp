FROM python:3.9-slim

WORKDIR /app

# System dependencies for python packages like xgboost or shap
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (ignoring the huge torch packages to save space if needed, 
# but installing all from requirements to ensure it runs)
# Note: You might want to use a CPU-only version of PyTorch for deployment to reduce size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port for the Flask app
EXPOSE 5000

# Start server using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "main:app"]
