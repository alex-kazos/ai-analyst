#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display help
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Available commands:"
    echo "  up           Start all services in detached mode"
    echo "  down         Stop and remove all services"
    echo "  build        Build or rebuild services"
    echo "  logs         View output from containers"
    echo "  shell        Open a bash shell in the web container"
    echo "  shell-db     Open a psql shell in the database container"
    echo "  shell-redis  Open a redis-cli shell in the redis container"
    echo "  migrate      Run database migrations"
    echo "  createsuperuser  Create a superuser"
    echo "  collectstatic  Collect static files"
    echo "  test        Run tests"
    echo "  help         Show this help message"
    echo ""
}

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}Created .env file. Please edit it with your configuration.${NC}"
fi

# Load environment variables
set -a
source .env
set +a

# Function to run docker-compose commands
dc() {
    docker-compose "$@"
}

# Function to run manage.py commands
manage() {
    docker-compose exec -T web python manage.py "$@"
}

# Parse command
case "$1" in
    up)
        echo -e "${GREEN}Starting services...${NC}"
        docker-compose up -d
        ;;
    down)
        echo -e "${YELLOW}Stopping services...${NC}"
        docker-compose down
        ;;
    build)
        echo -e "${GREEN}Building services...${NC}"
        docker-compose build --no-cache
        ;;
    logs)
        docker-compose logs -f
        ;;
    shell)
        docker-compose exec web bash
        ;;
    shell-db)
        docker-compose exec db psql -U $DB_USER -d $DB_NAME
        ;;
    shell-redis)
        docker-compose exec redis redis-cli -a "$REDIS_PASSWORD"
        ;;
    migrate)
        echo -e "${GREEN}Running migrations...${NC}"
        manage migrate
        ;;
    createsuperuser)
        manage createsuperuser
        ;;
    collectstatic)
        echo -e "${GREEN}Collecting static files...${NC}"
        manage collectstatic --noinput
        ;;
    test)
        echo -e "${GREEN}Running tests...${NC}"
        manage test
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
