FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY sec_core_unified.py .
COPY repo_scanner.py .

# Environment variables for HF Spaces
ENV PORT=7860
EXPOSE 7860

# Run the unified server
CMD ["python", "sec_core_unified.py"]
