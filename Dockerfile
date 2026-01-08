# 1. Base image
FROM python:3.11-slim

# 2. Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Working directory
WORKDIR /app

# 5. Copy requirements first
COPY requirements.txt .

# 6. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel




# 7. Install remaining deps (NO torch here)
RUN pip install --no-cache-dir -r requirements.txt

# 8. Download NLTK data
RUN python -m nltk.downloader stopwords

# 9. copy flask app folder
COPY flask_app ./flask_app

# 10. Expose port
EXPOSE 8000

# 11. Run Flask app
CMD ["python", "flask_app/app.py"]
