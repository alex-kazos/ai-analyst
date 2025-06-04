# Dockerfile for Django AI Analyst Web App
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies for mysqlclient and others
RUN apt-get update && \
    apt-get install -y gcc default-libmysqlclient-dev pkg-config build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project files
COPY . /app/

# Expose port (Django default)
EXPOSE 8000

# Collect static files (optional, skip if not using staticfiles)
RUN python manage.py collectstatic --noinput || true

# Run migrations (optional for dev)
# RUN python manage.py migrate || true

# Start server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
