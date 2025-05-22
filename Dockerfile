# Use an official Python runtime as a parent image
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install wait-for-it script
RUN curl -sSL https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh -o /usr/local/bin/wait-for-it \
    && chmod +x /usr/local/bin/wait-for-it \
    && mkdir -p /home/myuser/bin \
    && cp /usr/local/bin/wait-for-it /home/myuser/bin/ \
    && chown -R myuser:myuser /home/myuser/bin \
    && chmod +x /home/myuser/bin/wait-for-it

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Create a non-root user and switch to it
RUN useradd -m -d /home/myuser -s /bin/bash myuser

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/home/myuser/.local/bin:${PATH}" \
    PYTHONPATH="/home/myuser/app"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and configure the non-root user
RUN useradd -m -d /home/myuser -s /bin/bash myuser

# Set the working directory
WORKDIR /home/myuser/app

# Copy Python dependencies and wait script from builder
COPY --from=builder /root/.local /home/myuser/.local
COPY --from=builder /usr/local/bin/wait-for-it /usr/local/bin/wait-for-it

# Copy requirements first to leverage Docker cache
COPY --chown=myuser:myuser requirements.txt .

# Copy the rest of the application
COPY --chown=myuser:myuser . .


# Create necessary directories with proper ownership
RUN mkdir -p /home/myuser/app/staticfiles /home/myuser/app/media \
    && chown -R myuser:myuser /home/myuser/app \
    && chmod +x /usr/local/bin/wait-for-it

# Switch to non-root user
USER myuser

# Set environment variables for database and static files
ENV STATIC_ROOT=/home/myuser/app/staticfiles \
    MEDIA_ROOT=/home/myuser/app/media \
    WAIT_FOR_HOSTS=db:5432,redis:6379 \
    WAIT_FOR_TIMEOUT=60 \
    WAIT_FOR_RETRY_INTERVAL=5 \
    PATH="/home/myuser/bin:${PATH}"

# Collect static files
RUN python manage.py collectstatic --noinput --clear

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Create a startup script
RUN echo '#!/bin/sh\n\
# Wait for services to be ready\n\
echo "Waiting for services to be ready..."\n\
echo "WAIT_FOR_HOSTS: $WAIT_FOR_HOSTS"\n\
# Split the hosts by comma and wait for each one\
IFS=","\nfor hostport in $WAIT_FOR_HOSTS; do\n  host=$(echo $hostport | cut -d: -f1)\
  port=$(echo $hostport | cut -d: -f2)\
  echo "Waiting for $host:$port..."\n  wait-for-it $host:$port -t $WAIT_FOR_TIMEOUT -s -- echo "$host:$port is available"\ndone\n\n# Run database migrations\necho "Running database migrations..."\npython manage.py migrate --noinput\n\n# Start Gunicorn\necho "Starting Gunicorn..."\nexec gunicorn --bind 0.0.0.0:8000 \
         --workers 3 \
         --log-level=info \
         --access-logfile - \
         --error-logfile - \
         --capture-output \
         ai_analyst.wsgi:application\n' > /home/myuser/app/start.sh && \
    chmod +x /home/myuser/app/start.sh

# Command to run the application
CMD ["/home/myuser/app/start.sh"]
